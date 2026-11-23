"""
Policy Service: Manages policy ingestion, passage indexing, rule drafting, review, and approval.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

from reasoning.rule_schema import AcademicRule, CourseDefinition
from ingestion.pdf_parser import PDFParser
from ingestion.chunker import PolicyChunker
from ingestion.source_tracker import PolicyPassage, SourceTracker
from retrieval.embeddings import PolicyEmbedder
from retrieval.vector_store import AcademicVectorStore
from llm.rule_extractor import LLMRuleExtractor

DATA_DIR = Path(__file__).parent.parent / "data"
POLICIES_FILE = DATA_DIR / "policies.json"
RAW_POLICIES_DIR = DATA_DIR / "raw_policies"


class PolicyService:
    """Coordinates policy ingestion, draft rule review, and Knowledge Base updates."""

    def __init__(self):
        self.chunker = PolicyChunker()
        self.vector_store = AcademicVectorStore()
        self.embedder = PolicyEmbedder()
        self.extractor = LLMRuleExtractor()
        self.tracker = SourceTracker()
        self._load_approved_policies()

    def _load_approved_policies(self):
        if POLICIES_FILE.exists():
            with open(POLICIES_FILE, "r", encoding="utf-8") as f:
                self.policy_data = json.load(f)
        else:
            self.policy_data = {
                "institution": "Northbridge Institute of Technology (NITech)",
                "document_reference": "NIT-ACAD-REG-2026-V1",
                "courses": [],
                "rules": [],
            }

    def get_approved_rules(self) -> List[AcademicRule]:
        return [AcademicRule.model_validate(r) for r in self.policy_data.get("rules", [])]

    def get_courses(self) -> List[CourseDefinition]:
        return [CourseDefinition.model_validate(c) for c in self.policy_data.get("courses", [])]

    def process_uploaded_pdf(self, uploaded_file_bytes: bytes, filename: str) -> List[PolicyPassage]:
        """Saves uploaded PDF to raw_policies, extracts text passages, and indexes into vector store."""
        RAW_POLICIES_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = RAW_POLICIES_DIR / filename
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file_bytes)

        # Chunk PDF into passages
        passages = self.chunker.chunk_pdf(pdf_path)

        # Generate embeddings and store in vector database
        texts = [p.text for p in passages]
        embeddings = self.embedder.embed_batch(texts)
        self.vector_store.upsert_passages(passages, embeddings)

        for p in passages:
            self.tracker.register_passage(p)

        return passages

    def extract_draft_rule(self, passage: PolicyPassage) -> Optional[AcademicRule]:
        """Uses LLM to extract a draft rule from a passage."""
        return self.extractor.extract_rule_from_passage(passage)

    def approve_or_update_rule(self, rule: AcademicRule):
        """Appends or updates an approved rule in policies.json."""
        rule_dict = rule.model_dump()
        rule_dict["status"] = "approved"

        existing_rules = self.policy_data.setdefault("rules", [])
        updated = False
        for idx, r in enumerate(existing_rules):
            if r["id"] == rule.id:
                existing_rules[idx] = rule_dict
                updated = True
                break

        if not updated:
            existing_rules.append(rule_dict)

        # Save to disk
        with open(POLICIES_FILE, "w", encoding="utf-8") as f:
            json.dump(self.policy_data, f, indent=2)

    def delete_rule(self, rule_id: str):
        """Removes a rule from policies.json."""
        existing_rules = self.policy_data.get("rules", [])
        self.policy_data["rules"] = [r for r in existing_rules if r["id"] != rule_id]
        with open(POLICIES_FILE, "w", encoding="utf-8") as f:
            json.dump(self.policy_data, f, indent=2)
