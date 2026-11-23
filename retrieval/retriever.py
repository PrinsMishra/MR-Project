"""
Hybrid Retriever for Academic Policy Documents.
Combines semantic vector search with keyword/course-code filtering.
"""

from typing import List, Optional
from retrieval.embeddings import PolicyEmbedder
from retrieval.vector_store import AcademicVectorStore, SearchResult


class PolicyRetriever:
    """Retrieves relevant policy passages for student eligibility queries."""

    def __init__(self, vector_store: Optional[AcademicVectorStore] = None):
        self.vector_store = vector_store or AcademicVectorStore()
        self.embedder = PolicyEmbedder()

    def retrieve(
        self,
        query: str,
        course_code: Optional[str] = None,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Retrieves top policy passages relevant to the user query.
        Applies semantic similarity search augmented with course-aware and keyword boosting.
        """
        q_vec = self.embedder.embed_text(query)
        results = self.vector_store.query(q_vec, top_k=100)

        # Hybrid ranking: semantic score + course relevance + domain keyword matching
        boosted = []
        q_lower = query.lower()

        for r in results:
            score = r.score
            # Strong boost if the passage directly mentions the target course
            if course_code and (course_code in r.course_hints or course_code in r.text):
                score += 1.0

            # Boost for policy topic keywords
            for keyword in ["credit", "waiver", "prerequisite", "approval", "probation", "hold", "repeat"]:
                if keyword in q_lower and keyword in r.text.lower():
                    score += 0.4

            boosted.append((score, r))

        boosted.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in boosted[:top_k]]

    def format_context(self, search_results: List[SearchResult]) -> str:
        """Formats retrieved passages into a structured markdown context string."""
        if not search_results:
            return "No matching policy passages found."

        chunks = []
        for r in search_results:
            header = f"[{r.passage_id}] {r.document} (Section {r.section}, Page {r.page_number})"
            chunks.append(f"{header}:\n\"{r.text}\"")

        return "\n\n".join(chunks)
