"""
Test Suite for Phase 1: Policy Dataset, Rule Schema, and Test Cases Validation.
"""

import json
from pathlib import Path
import pytest

from reasoning.rule_schema import (
    AcademicRule,
    StudentProfile,
    TestCase,
    CourseDefinition,
    DecisionEnum,
)

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def policies_data():
    with open(DATA_DIR / "policies.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def students_data():
    with open(DATA_DIR / "students.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def test_cases_data():
    with open(DATA_DIR / "test_cases.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_policies_schema_and_integrity(policies_data):
    """Test that all 25 rules adhere to AcademicRule schema and have unique IDs."""
    rules = policies_data["rules"]
    assert len(rules) == 25, f"Expected exactly 25 rules, found {len(rules)}"

    rule_ids = set()
    for rule_raw in rules:
        rule = AcademicRule.model_validate(rule_raw)
        assert rule.id.startswith("R"), f"Rule ID must start with R: {rule.id}"
        assert rule.id not in rule_ids, f"Duplicate rule ID: {rule.id}"
        rule_ids.add(rule.id)
        assert rule.source.document != ""
        assert rule.source.section != ""
        assert rule.source.passage_id != ""

    # Check rule IDs R001 through R025 are present
    expected_ids = {f"R{i:03d}" for i in range(1, 26)}
    assert rule_ids == expected_ids, f"Missing rules: {expected_ids - rule_ids}"


def test_courses_schema_and_prerequisites_dag(policies_data):
    """Test course definitions and ensure prerequisite graph is acyclic."""
    courses_raw = policies_data["courses"]
    courses = [CourseDefinition.model_validate(c) for c in courses_raw]
    course_codes = {c.code for c in courses}
    assert {"CS101", "CS201", "CS301", "CS401", "CS402"}.issubset(course_codes)

    # Build dependency graph
    prereq_graph = {c.code: set() for c in courses}
    for rule in policies_data["rules"]:
        if rule["type"] == "prerequisite" and rule.get("prerequisites"):
            for target in rule["target_courses"]:
                for prereq in rule["prerequisites"]:
                    code = prereq["course_code"]
                    assert code in course_codes, f"Prerequisite {code} not in course catalog"
                    prereq_graph[target].add(code)

    # Topological sort / cycle detection
    visited = {}

    def dfs(node, path):
        visited[node] = True
        for neighbor in prereq_graph.get(node, []):
            if neighbor in path:
                raise ValueError(f"Cycle detected in prerequisites: {path} -> {neighbor}")
            if not visited.get(neighbor, False):
                dfs(neighbor, path + [neighbor])

    for c in course_codes:
        dfs(c, [c])


def test_students_schema_and_diversity(students_data):
    """Test that all student profiles validate against StudentProfile schema."""
    students = students_data["students"]
    assert len(students) >= 15, f"Expected at least 15 students, found {len(students)}"

    student_ids = set()
    has_waiver_student = False
    has_unknown_fact_student = False

    for student_raw in students:
        student = StudentProfile.model_validate(student_raw)
        assert student.student_id not in student_ids, f"Duplicate student ID: {student.student_id}"
        student_ids.add(student.student_id)

        if student.faculty_waivers:
            has_waiver_student = True
        if student.cgpa is None or student.year is None or student.faculty_waivers is None:
            has_unknown_fact_student = True

    assert has_waiver_student, "Dataset must include at least one student with a faculty waiver"
    assert has_unknown_fact_student, "Dataset must include students with missing/unknown facts"


def test_test_cases_cross_referencing(policies_data, students_data, test_cases_data):
    """Test that test cases cross-reference valid rules, students, and courses."""
    rules_dict = {r["id"]: r for r in policies_data["rules"]}
    students_dict = {s["student_id"]: s for s in students_data["students"]}
    courses_dict = {c["code"]: c for c in policies_data["courses"]}

    test_cases = test_cases_data["test_cases"]
    assert len(test_cases) >= 15

    decisions = set()
    for tc_raw in test_cases:
        tc = TestCase.model_validate(tc_raw)
        decisions.add(tc.expected_decision)

        # Cross-reference student
        assert tc.student_id in students_dict, f"Unknown student {tc.student_id} in {tc.test_id}"

        # Cross-reference course
        assert tc.target_course in courses_dict, f"Unknown course {tc.target_course} in {tc.test_id}"

        # Cross-reference failing rules
        for r_id in tc.expected_failing_rules:
            assert r_id in rules_dict, f"Unknown rule {r_id} referenced in {tc.test_id}"

    assert DecisionEnum.ELIGIBLE in decisions
    assert DecisionEnum.INELIGIBLE in decisions
    assert DecisionEnum.UNDETERMINED in decisions

