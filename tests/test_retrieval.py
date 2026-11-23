"""
Unit and Integration Tests for RAG Passage Retrieval.
"""

import pytest
from retrieval.retriever import PolicyRetriever
from retrieval.vector_store import AcademicVectorStore


@pytest.fixture(scope="module")
def retriever():
    store = AcademicVectorStore()
    return PolicyRetriever(vector_store=store)


def test_retrieval_for_cs301(retriever):
    """Verifies that asking about CS301 retrieves the prerequisite and CGPA rules."""
    query = "Am I eligible to register for Machine Learning CS301?"
    results = retriever.retrieve(query, course_code="CS301", top_k=5)
    assert len(results) > 0

    p_ids = [r.passage_id for r in results]
    # P003 is Rule R003 (CS301 prerequisites), P008 is Rule R008 (CS301 CGPA)
    assert any(pid in p_ids for pid in ["P003", "P008", "P016"]), (
        f"Expected CS301 passages in top results, got {p_ids}"
    )


def test_retrieval_for_credit_overload(retriever):
    """Verifies that asking about 28 credits retrieves overload exception rules."""
    query = "Can I take 28 credits this semester with department approval?"
    results = retriever.retrieve(query, top_k=5)
    assert len(results) > 0

    p_ids = [r.passage_id for r in results]
    # P013 is R013 (28 credit exception), P014 is R014 (28 credit ceiling), P012 is R012 (24 credit standard)
    assert any(pid in p_ids for pid in ["P012", "P013", "P014"]), (
        f"Expected credit limit passages, got {p_ids}"
    )


def test_retrieval_for_prerequisite_waiver(retriever):
    """Verifies that asking about waiver retrieves faculty waiver policies."""
    query = "Can I register without completing a prerequisite if I have faculty approval?"
    results = retriever.retrieve(query, top_k=5)
    assert len(results) > 0

    p_ids = [r.passage_id for r in results]
    # P006 is R006 (waiver policy), P021 is R021 (waiver specificity)
    assert any(pid in p_ids for pid in ["P006", "P021", "P002", "P003"]), (
        f"Expected waiver passages, got {p_ids}"
    )
