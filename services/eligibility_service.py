"""
Eligibility Service: Integrates RAG Policy Retrieval with Z3 Formal Verification.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from reasoning.rule_schema import StudentProfile, AcademicRule, DecisionEnum
from reasoning.z3_engine import Z3Reasoner, EligibilityResult
from reasoning.counterexample import CounterexampleGenerator, CounterexampleResult
from retrieval.retriever import PolicyRetriever
from retrieval.vector_store import SearchResult


class VerificationBundle(BaseModel):
    decision: DecisionEnum
    target_course: str
    student_id: str
    student_name: str
    explanation: str
    retrieved_passages: List[SearchResult] = []
    failing_rules: List[str] = []
    failing_rule_details: List[Dict[str, Any]] = []
    missing_facts: List[str] = []
    counterexample_model: Optional[Dict[str, Any]] = None
    hypothesis_counterexample: Optional[str] = None


class EligibilityService:
    """End-to-End Orchestrator for RAG Retrieval and Z3 Verification."""

    def __init__(self):
        self.retriever = PolicyRetriever()
        self.reasoner = Z3Reasoner()
        self.counterexample_gen = CounterexampleGenerator()

    def verify_registration(
        self,
        student: StudentProfile,
        target_course: str,
        user_query: Optional[str] = None,
    ) -> VerificationBundle:
        """
        1. Retrieves relevant policy passages using RAG.
        2. Formulates and solves formal constraints using Z3.
        3. Identifies failing rules, unsat cores, missing facts, and counterexamples.
        """
        query_text = user_query or f"Registration eligibility requirements for {target_course}"
        retrieved_passages = self.retriever.retrieve(query_text, course_code=target_course, top_k=4)

        # Execute Z3 formal evaluation
        z3_res = self.reasoner.evaluate(student, target_course)

        # Assemble details for failing rules with exact source citations
        failing_details = []
        for rid in z3_res.failing_rules:
            r_obj = self.reasoner.rules_lookup.get(rid)
            if r_obj:
                failing_details.append({
                    "rule_id": rid,
                    "title": r_obj.title,
                    "description": r_obj.description,
                    "source_doc": r_obj.source.document,
                    "source_section": r_obj.source.section,
                    "passage_id": r_obj.source.passage_id,
                    "source_text": r_obj.source.text,
                })

        # Check for counterexamples if claim or boundary case
        ce_model = z3_res.counterexample
        hyp_ce_str = None
        if z3_res.decision == DecisionEnum.UNDETERMINED and not ce_model:
            # Check high GPA fallacy counterexample for illustration
            ce_res = self.counterexample_gen.check_claim_high_gpa_implies_eligible(target_course)
            if ce_res.found:
                hyp_ce_str = ce_res.explanation

        return VerificationBundle(
            decision=z3_res.decision,
            target_course=target_course,
            student_id=student.student_id,
            student_name=student.name,
            explanation=z3_res.explanation,
            retrieved_passages=retrieved_passages,
            failing_rules=z3_res.failing_rules,
            failing_rule_details=failing_details,
            missing_facts=z3_res.missing_facts,
            counterexample_model=ce_model,
            hypothesis_counterexample=hyp_ce_str,
        )
