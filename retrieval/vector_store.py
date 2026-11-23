"""
Vector Store for Academic Policy Passages.
Supports Pinecone with automatic fallback to local vector index
for zero-friction local execution.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel
from dotenv import load_dotenv

from ingestion.source_tracker import PolicyPassage

load_dotenv()


class VectorRecord(BaseModel):
    id: str
    values: List[float]
    metadata: Dict[str, Any]


class SearchResult(BaseModel):
    passage_id: str
    document: str
    section: str
    page_number: int
    text: str
    score: float
    rule_hints: List[str] = []
    course_hints: List[str] = []


class AcademicVectorStore:
    """Hybrid Vector Store (Pinecone Cloud + Local Persistence)."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "medicalindex")
        self.namespace = "academic_policies"

        self.storage_dir = storage_dir or (Path(__file__).parent.parent / "data" / "vector_cache")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.storage_dir / "passages_index.json"

        self.local_records: Dict[str, VectorRecord] = {}
        self._load_local_cache()

        # Initialize Pinecone if reachable with short timeout
        self.pinecone_index = None
        if self.pinecone_api_key:
            try:
                import socket
                s = socket.create_connection(("api.pinecone.io", 443), timeout=0.3)
                s.close()
                from pinecone import Pinecone
                pc = Pinecone(api_key=self.pinecone_api_key)
                existing = [idx["name"] for idx in pc.list_indexes()]
                if self.index_name in existing:
                    self.pinecone_index = pc.Index(self.index_name)
            except Exception:
                self.pinecone_index = None

    def upsert_passages(self, passages: List[PolicyPassage], embeddings: List[List[float]]):
        """Stores policy passages and their dense embeddings."""
        pinecone_vectors = []

        for passage, emb in zip(passages, embeddings):
            meta = {
                "passage_id": passage.passage_id,
                "document": passage.document,
                "section": passage.section,
                "page_number": passage.page_number,
                "text": passage.text,
                "rule_hints": passage.rule_hints,
                "course_hints": passage.course_hints,
            }
            rec = VectorRecord(id=passage.passage_id, values=emb, metadata=meta)
            self.local_records[passage.passage_id] = rec

            if self.pinecone_index:
                pinecone_vectors.append({
                    "id": passage.passage_id,
                    "values": emb,
                    "metadata": meta,
                })

        # Persist locally
        self._save_local_cache()

        # Push to Pinecone if connected
        if self.pinecone_index and pinecone_vectors:
            try:
                self.pinecone_index.upsert(vectors=pinecone_vectors, namespace=self.namespace)
            except Exception:
                pass

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        course_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Queries the vector store with cosine similarity and optional course filtering."""
        results = []

        # Local cosine similarity search
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1.0

        for pid, rec in self.local_records.items():
            meta = rec.metadata

            # Apply course filter if specified
            if course_filter and course_filter not in meta.get("course_hints", []):
                # Check if text mentions course directly
                if course_filter not in meta.get("text", ""):
                    continue

            d_vec = np.array(rec.values, dtype=np.float32)
            d_norm = np.linalg.norm(d_vec)
            if d_norm == 0:
                d_norm = 1.0

            cos_sim = float(np.dot(q_vec, d_vec) / (q_norm * d_norm))

            results.append(SearchResult(
                passage_id=pid,
                document=meta["document"],
                section=meta["section"],
                page_number=meta["page_number"],
                text=meta["text"],
                score=cos_sim,
                rule_hints=meta.get("rule_hints", []),
                course_hints=meta.get("course_hints", []),
            ))

        # Sort descending by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _save_local_cache(self):
        data = {pid: rec.model_dump() for pid, rec in self.local_records.items()}
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_local_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.local_records = {pid: VectorRecord.model_validate(rec) for pid, rec in data.items()}
            except Exception:
                self.local_records = {}
