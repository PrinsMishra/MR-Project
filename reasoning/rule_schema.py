"""
Rule Schema & Data Models for Academic Policy Reasoning.

Defines Pydantic models for:
- Source references (traceability back to policy documents)
- Conditions (atomic and compound logical conditions)
- Academic Rules (R001 - R025)
- Course Definitions
- Student Profiles (supporting known and unknown/missing facts)
- Test Cases (with ground-truth decisions: ELIGIBLE, INELIGIBLE, UNDETERMINED)
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class DecisionEnum(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    UNDETERMINED = "UNDETERMINED"
    CONFLICT = "CONFLICT"


class RuleType(str, Enum):
    PREREQUISITE = "prerequisite"
    MINIMUM_CGPA = "minimum_cgpa"
    PROBATION_RESTRICTION = "probation_restriction"
    CREDIT_LIMIT = "credit_limit"
    CREDIT_EXCEPTION = "credit_exception"
    YEAR_STANDING = "year_standing"
    REPEAT_RESTRICTION = "repeat_restriction"
    COURSE_OFFERED = "course_offered"
    ADMIN_HOLD = "admin_hold"
    WAIVER_POLICY = "waiver_policy"
    APPROVAL_POLICY = "approval_policy"
    INTEGRITY_INVARIANT = "integrity_invariant"


class ComparisonOperator(str, Enum):
    EQ = "=="
    NEQ = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


class SourceReference(BaseModel):
    document: str = Field(..., description="Filename or title of source document")
    section: str = Field(..., description="Section number, e.g. '1.2'")
    passage_id: str = Field(..., description="Unique passage ID, e.g. 'P003'")
    text: Optional[str] = Field(None, description="Exact text excerpt from the policy")


class AtomicCondition(BaseModel):
    field: str = Field(..., description="Student or course attribute name (e.g., 'cgpa', 'year')")
    operator: ComparisonOperator = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Threshold or target value")


class CompoundCondition(BaseModel):
    logical_op: Literal["AND", "OR", "NOT"] = Field("AND", description="Logical combinator")
    conditions: List[Union[AtomicCondition, CompoundCondition]] = Field(
        ..., description="Nested sub-conditions"
    )


Condition = Union[AtomicCondition, CompoundCondition]


class CoursePrereqItem(BaseModel):
    course_code: str
    waiver_allowed: bool = True


class AcademicRule(BaseModel):
    id: str = Field(..., description="Rule identifier, e.g. 'R001'")
    type: RuleType = Field(..., description="Classification category of the rule")
    title: str = Field(..., description="Short human-readable title")
    description: str = Field(..., description="Plain-English explanation of the policy")
    target_courses: List[str] = Field(
        default_factory=list,
        description="Course codes this rule restricts (empty list means applies globally to all courses)",
    )
    # Structured condition for Z3 compilation
    condition: Optional[Union[AtomicCondition, CompoundCondition]] = Field(
        None, description="Formal condition that must be satisfied for eligibility"
    )
    # Prerequisite-specific specification
    prerequisites: Optional[List[CoursePrereqItem]] = Field(
        None, description="List of required prerequisite courses if rule_type is prerequisite"
    )
    # Waiver specifications
    waiver_allowed: bool = Field(False, description="Whether a faculty waiver can override this requirement")
    waiver_type: Optional[str] = Field(None, description="e.g. 'faculty_waiver', 'dept_approval'")
    source: SourceReference
    status: Literal["approved", "draft", "rejected"] = Field("approved", description="Human review status")


class CourseDefinition(BaseModel):
    code: str
    title: str
    credits: int
    is_offered: bool = True


class PrerequisiteWaiver(BaseModel):
    course_code: str = Field(..., description="The course for which waiver is requested, e.g. 'CS301'")
    waived_prerequisite: str = Field(..., description="The prerequisite waived, e.g. 'CS201'")
    approved_by: str = Field(..., description="Faculty instructor name or 'Faculty'")


class StudentProfile(BaseModel):
    student_id: str
    name: str
    year: Optional[int] = Field(None, description="1, 2, 3, or 4. None if unknown.")
    cgpa: Optional[float] = Field(None, description="CGPA on 10.0 scale. None if unknown.")
    current_registered_credits: Optional[int] = Field(
        0, description="Credits currently enrolled in this semester"
    )
    completed_courses: Optional[List[str]] = Field(
        default_factory=list, description="Courses completed in prior terms. None if unknown."
    )
    on_probation: Optional[bool] = Field(False, description="Academic probation status. None if unknown.")
    registration_hold: Optional[bool] = Field(
        False, description="True if student has an administrative hold. None if unknown."
    )
    department_credit_approval: Optional[bool] = Field(
        False, description="Department chair approval for >24 credits. None if unknown."
    )
    faculty_waivers: Optional[List[PrerequisiteWaiver]] = Field(
        default_factory=list, description="Explicit faculty waivers granted to this student"
    )

    def has_waiver_for(self, target_course: str, prerequisite_course: str) -> bool:
        if not self.faculty_waivers:
            return False
        return any(
            w.course_code == target_course and w.waived_prerequisite == prerequisite_course
            for w in self.faculty_waivers
        )


class TestCase(BaseModel):
    __test__ = False
    test_id: str
    name: str
    description: str
    student_id: str
    target_course: str
    expected_decision: DecisionEnum
    expected_failing_rules: List[str] = Field(
        default_factory=list, description="Rule IDs expected to fail (for INELIGIBLE)"
    )
    expected_missing_facts: List[str] = Field(
        default_factory=list, description="Fields required to decide (for UNDETERMINED)"
    )
    notes: Optional[str] = None
