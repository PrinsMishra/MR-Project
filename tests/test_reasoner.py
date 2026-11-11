"""
Reasoning Engine Unit & Regression Tests.
Verifies all 15 benchmark test cases against the Z3 Formal Solver.
"""

import json
from pathlib import Path
import pytest

from reasoning.rule_schema import StudentProfile, TestCase, DecisionEnum
from reasoning.z3_engine import Z3Reasoner

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def reasoner():
    return Z3Reasoner(DATA_DIR / "policies.json")


@pytest.fixture(scope="module")
def students_map():
    with open(DATA_DIR / "students.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {s["student_id"]: StudentProfile.model_validate(s) for s in data["students"]}


@pytest.fixture(scope="module")
def test_cases():
    with open(DATA_DIR / "test_cases.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return [TestCase.model_validate(tc) for tc in data["test_cases"]]


def test_all_15_baseline_scenarios(reasoner, students_map, test_cases):
    """Executes all 15 test cases and asserts the decision matches ground truth."""
    for tc in test_cases:
        student = students_map[tc.student_id]
        result = reasoner.evaluate(student, tc.target_course)

        assert result.decision == tc.expected_decision, (
            f"Test case {tc.test_id} ({tc.name}) failed! "
            f"Expected {tc.expected_decision}, got {result.decision}. "
            f"Explanation: {result.explanation}"
        )

        if tc.expected_decision == DecisionEnum.INELIGIBLE and tc.expected_failing_rules:
            for expected_rule in tc.expected_failing_rules:
                assert expected_rule in result.failing_rules, (
                    f"Test case {tc.test_id}: expected rule {expected_rule} to fail, "
                    f"but failing rules were {result.failing_rules}"
                )

        if tc.expected_decision == DecisionEnum.UNDETERMINED and tc.expected_missing_facts:
            for missing_fact in tc.expected_missing_facts:
                assert missing_fact in result.missing_facts, (
                    f"Test case {tc.test_id}: expected missing fact {missing_fact}, "
                    f"but missing facts were {result.missing_facts}"
                )

