"""
Interactive Demo Script for Academic Policy Reasoning Engine.
Usage: python demo.py [student_id] [course_code]
Example: python demo.py S001 CS301
"""

import sys
import json
from pathlib import Path
from reasoning.rule_schema import StudentProfile
from reasoning.z3_engine import Z3Reasoner

DATA_DIR = Path(__file__).parent / "data"


def main():
    reasoner = Z3Reasoner(DATA_DIR / "policies.json")

    with open(DATA_DIR / "students.json", "r", encoding="utf-8") as f:
        students_data = json.load(f)["students"]
    students_map = {s["student_id"]: StudentProfile.model_validate(s) for s in students_data}

    student_id = sys.argv[1] if len(sys.argv) > 1 else "S003"
    target_course = sys.argv[2] if len(sys.argv) > 2 else "CS301"

    if student_id not in students_map:
        print(f"Unknown student: {student_id}. Available: {list(students_map.keys())}")
        return

    student = students_map[student_id]
    print("=" * 70)
    print(f"  Counterexample-Guided Academic Policy Reasoning with LLMs and Z3")
    print("=" * 70)
    print(f"Student: {student.name} ({student.student_id})")
    print(f"Year: {student.year} | CGPA: {student.cgpa} | Current Credits: {student.current_registered_credits}")
    print(f"Completed Courses: {student.completed_courses}")
    if student.faculty_waivers:
        for w in student.faculty_waivers:
            print(f"Faculty Waiver: Waived {w.waived_prerequisite} for {w.course_code} (by {w.approved_by})")
    print(f"Target Registration: {target_course}")
    print("-" * 70)

    result = reasoner.evaluate(student, target_course)
    print(f"DECISION: {result.decision.value}")
    print(f"EXPLANATION:\n  {result.explanation}")

    if result.failing_rules:
        print("\nFAILING POLICIES:")
        for r in result.failing_rules:
            desc = reasoner.rules_lookup[r].description
            src = reasoner.rules_lookup[r].source
            print(f"  - [{r}] {desc} (Ref: {src.document}, Sec {src.section}, {src.passage_id})")

    if result.missing_facts:
        print(f"\nMISSING FACTS: {result.missing_facts}")

    if result.source_references:
        print("\nSUPPORTING SOURCES CITED:")
        for s in result.source_references[:3]:
            print(f"  - [{s.passage_id}] {s.document} §{s.section}")

    print("=" * 70)


if __name__ == "__main__":
    main()

