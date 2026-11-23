"""
Unit & Regression Tests for Counterexample Generation and Conflict Detection (Phase 4).
Tests 5 distinct counterexample scenarios and policy inconsistency detection.
"""

from pathlib import Path
import pytest
import z3

from reasoning.counterexample import CounterexampleGenerator
from reasoning.conflict_detector import ConflictDetector
from reasoning.rule_schema import AcademicRule, AtomicCondition, RuleType, SourceReference

POLICIES_PATH = Path(__file__).parent.parent / "data" / "policies.json"


@pytest.fixture(scope="module")
def ce_gen():
    return CounterexampleGenerator(POLICIES_PATH)


@pytest.fixture(scope="module")
def conflict_det():
    return ConflictDetector(POLICIES_PATH)


# ---------------- Counterexample Scenario Tests ----------------

def test_counterexample_1_high_gpa_fallacy(ce_gen):
    """
    Claim: 'All students with CGPA >= 8.5 are eligible for CS301 (Machine Learning).'
    Expected: Falsified! A student with CGPA 9.0 who has not completed CS201 is ineligible.
    """
    res = ce_gen.check_claim_high_gpa_implies_eligible(target_course="CS301", min_gpa=8.5)
    assert res.found is True, "Expected counterexample for naive high GPA claim"
    assert res.counterexample is not None
    # Verify the counterexample model has high GPA but violates a prerequisite
    assert res.counterexample["cgpa"] >= 8.5
    assert res.violated_rule_id in ["R003", "R016", "R012", "R023", "R019"]


def test_counterexample_2_prerequisites_alone_fallacy(ce_gen):
    """
    Claim: 'Any student who completed CS101 and CS201 is eligible for CS301.'
    Expected: Falsified! A student with completed prerequisites but CGPA < 7.0 is ineligible.
    """
    res = ce_gen.check_claim_prerequisites_alone_imply_eligible(target_course="CS301")
    assert res.found is True
    assert res.counterexample is not None
    # In counterexample, completed_CS101 and completed_CS201 are True, but CGPA or standing fails
    assert res.counterexample.get("completed_CS101") is True
    assert res.counterexample.get("completed_CS201") is True


def test_counterexample_3_approval_alone_fallacy(ce_gen):
    """
    Claim: 'Any student with department approval can register for 28 credits in CS401.'
    Expected: Falsified! A Year 2 or 3 student (or CGPA < 8.0) cannot take 28 credits despite approval.
    """
    res = ce_gen.check_claim_approval_alone_grants_28_credits(target_course="CS401")
    assert res.found is True
    assert res.counterexample is not None
    assert res.counterexample.get("department_credit_approval") is True
    assert res.counterexample.get("total_credits") == 28


def test_counterexample_4_waiver_leakage_fallacy(ce_gen):
    """
    Claim: 'A faculty waiver granted for CS201/CS101 satisfies prerequisites for CS301.'
    Expected: Falsified! Under Rule R021, waivers are course-specific and do not leak to CS301.
    """
    res = ce_gen.check_claim_waiver_for_other_course_applies(
        target_course="CS301", waived_course="CS201", prereq_code="CS101"
    )
    assert res.found is True
    assert res.counterexample is not None


def test_counterexample_5_standing_probation_fallacy(ce_gen):
    """
    Claim: 'Any student with CGPA >= 8.0 not on probation is eligible for CS401.'
    Expected: Falsified! Year 1 or Year 2 students are ineligible (violates R017: Year >= 3).
    """
    res = ce_gen.check_claim_good_standing_implies_elective_eligible(target_course="CS401")
    assert res.found is True
    assert res.counterexample is not None


# ---------------- Policy Contradiction Tests ----------------

def test_approved_policy_set_is_internally_consistent(conflict_det):
    """Verifies that the ground-truth approved policy set has NO internal contradictions."""
    for c in ["CS101", "CS201", "CS301", "CS401", "CS402"]:
        rep = conflict_det.check_policy_set_consistency(target_course=c)
        assert rep.has_conflict is False, f"Unexpected contradiction found in approved policies for {c}"


def test_conflict_detection_on_contradictory_candidate_rule(conflict_det):
    """
    Simulates injecting an contradictory candidate rule:
    'No student may register for CS301 if their CGPA is greater than 6.0.'
    Since R008 requires CGPA >= 7.0, no student can ever have CGPA >= 7.0 and CGPA <= 6.0 simultaneously!
    Expected: ConflictDetector detects UNSAT core containing R008 and candidate rule.
    """
    conflicting_rule = AcademicRule(
        id="R_CANDIDATE_CONFLICT",
        type=RuleType.MINIMUM_CGPA,
        title="Contradictory Low CGPA Ceiling",
        description="Students with CGPA > 6.0 are prohibited from registering for CS301.",
        target_courses=["CS301"],
        condition=AtomicCondition(
            field="cgpa",
            operator="<=",
            value=6.0,
        ),
        source=SourceReference(
            document="unverified_memo.pdf",
            section="1.0",
            passage_id="P_UNV",
            text="Simulated contradictory draft rule",
        ),
        status="draft",
    )

    rep = conflict_det.test_candidate_rule_conflict(conflicting_rule, target_course="CS301")
    assert rep.has_conflict is True, "Expected conflict detector to catch contradictory rule"
    assert "R_CANDIDATE_CONFLICT" in rep.conflicting_rule_ids
    assert "R008" in rep.conflicting_rule_ids

