"""
Interactive RAG + Z3 Demonstration Script.
Accepts student questions, retrieves supporting academic policy passages,
evaluates student facts using Z3, and returns grounded, explainable answers.

Usage:
  python demo_rag.py "Am I eligible to register for Machine Learning?" --student S003 --course CS301
  python demo_rag.py "Can I take 28 credits this semester?" --student S008
  python demo_rag.py "Which academic rule prevents my registration?" --student S002 --course CS301
"""

import sys
import argparse
import json
from pathlib import Path

from retrieval.vector_store import AcademicVectorStore
from retrieval.retriever import PolicyRetriever
from reasoning.rule_schema import StudentProfile
from reasoning.z3_engine import Z3Reasoner

DATA_DIR = Path(__file__).parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Academic Policy RAG + Z3 Assistant")
    parser.add_argument("question", nargs="?", default="Am I eligible to register for Machine Learning?", help="Student question")
    parser.add_argument("--student", default="S003", help="Student ID from students.json (e.g. S001, S002, S003, S008)")
    parser.add_argument("--course", default=None, help="Target course code (e.g. CS301, CS401)")

    args = parser.parse_args()

    # Determine course from query if not specified
    target_course = args.course
    if not target_course:
        for c in ["CS101", "CS201", "CS301", "CS401", "CS402"]:
            if c in args.question.upper() or c in args.question:
                target_course = c
                break
        if not target_course and "machine learning" in args.question.lower():
            target_course = "CS301"
        elif not target_course and "deep learning" in args.question.lower():
            target_course = "CS401"
        elif not target_course and "data structures" in args.question.lower():
            target_course = "CS201"
        elif not target_course:
            target_course = "CS301"

    print("=" * 75)
    print("  COUNTEREXAMPLE-GUIDED ACADEMIC POLICY REASONING WITH LLMS AND Z3")
    print("=" * 75)
    print(f"Student Question: \"{args.question}\"")
    print(f"Target Course:    {target_course}")
    print(f"Student Profile:  {args.student}")
    print("-" * 75)

    # 1. RAG Retrieval Phase
    print("\n[Step 1: RAG Policy Retrieval]")
    retriever = PolicyRetriever()
    search_results = retriever.retrieve(args.question, course_code=target_course, top_k=4)

    print(f"Retrieved {len(search_results)} supporting policy passages from vector store:")
    for idx, r in enumerate(search_results, 1):
        print(f"  {idx}. [{r.passage_id}] {r.document} §{r.section} (Score: {r.score:.3f})")
        print(f"     \"{r.text}\"")

    # 2. Z3 Formal Verification Phase
    print("\n[Step 2: Z3 Formal Symbolic Verification]")
    with open(DATA_DIR / "students.json", "r", encoding="utf-8") as f:
        students_data = json.load(f)["students"]
    students_map = {s["student_id"]: StudentProfile.model_validate(s) for s in students_data}

    if args.student in students_map:
        student = students_map[args.student]
        print(f"Loaded facts for {student.name} ({student.student_id}):")
        print(f"  - Academic Year: {student.year} | CGPA: {student.cgpa} | Credits: {student.current_registered_credits}")
        print(f"  - Completed Courses: {student.completed_courses}")
        if student.faculty_waivers:
            for w in student.faculty_waivers:
                print(f"  - Faculty Waiver: Waived {w.waived_prerequisite} for {w.course_code}")

        reasoner = Z3Reasoner(DATA_DIR / "policies.json")
        result = reasoner.evaluate(student, target_course)

        print("\n[Step 3: Formal Verdict & Grounded Explanation]")
        print(f"Formal Decision: {result.decision.value}")
        print(f"Explanation:     {result.explanation}")

        if result.failing_rules:
            print("\nBlocking Policies (Identified via Z3 UNSAT Core):")
            for rid in result.failing_rules:
                r_obj = reasoner.rules_lookup.get(rid)
                if r_obj:
                    print(f"  ❌ [{rid}] {r_obj.description} (Source: {r_obj.source.passage_id})")

        if result.missing_facts:
            print(f"\nMissing Information Needed for Verdict: {result.missing_facts}")

    print("=" * 75)


if __name__ == "__main__":
    main()
