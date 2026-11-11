"""
Rule Compiler: Translates Structured JSON Academic Policies and Student Profiles into Z3 Constraints.
"""

from typing import Dict, List, Tuple, Any, Optional
import z3

from reasoning.rule_schema import (
    AcademicRule,
    StudentProfile,
    CourseDefinition,
)


class Z3PolicyContext:
    """Encapsulates the Z3 variables, compiled constraints, and rule tracking for a query."""

    def __init__(self, target_course: str, course_catalog: Dict[str, CourseDefinition]):
        self.target_course = target_course
        self.catalog = course_catalog

        # Core Z3 variables
        self.cgpa = z3.Real("cgpa")
        self.year = z3.Int("year")
        self.credits_current = z3.Int("credits_current")
        self.credits_requested = z3.Int("credits_requested")
        self.total_credits = z3.Int("total_credits")

        self.on_probation = z3.Bool("on_probation")
        self.registration_hold = z3.Bool("registration_hold")
        self.department_credit_approval = z3.Bool("department_credit_approval")

        self.course_offered = z3.Bool("course_offered")
        self.course_already_completed = z3.Bool("course_already_completed")

        # Dynamic course and waiver variables
        self.completed_vars: Dict[str, z3.BoolRef] = {}
        self.waiver_vars: Dict[str, z3.BoolRef] = {}

        # Structural constraints (facts, domain bounds, arithmetic totals)
        self.structural_constraints: List[z3.BoolRef] = []

        # Rule assertions: mapping rule_id -> (z3_condition_expression, label_bool)
        self.rule_assertions: Dict[str, Tuple[z3.BoolRef, z3.BoolRef]] = {}

        # Tracking unknown fields
        self.unknown_fields: List[str] = []

        self._init_structural_constraints()

    def get_completed_var(self, course_code: str) -> z3.BoolRef:
        if course_code not in self.completed_vars:
            self.completed_vars[course_code] = z3.Bool(f"completed_{course_code}")
        return self.completed_vars[course_code]

    def get_waiver_var(self, target_course: str, prereq_course: str) -> z3.BoolRef:
        key = f"{target_course}_waived_{prereq_course}"
        if key not in self.waiver_vars:
            self.waiver_vars[key] = z3.Bool(f"waiver_{key}")
        return self.waiver_vars[key]

    def _init_structural_constraints(self):
        """Domain constraints on GPA, academic year, and credit calculations."""
        req_credits = self.catalog[self.target_course].credits if self.target_course in self.catalog else 4
        self.structural_constraints.extend([
            self.credits_requested == req_credits,
            self.total_credits == self.credits_current + self.credits_requested,
            self.credits_current >= 0,
            self.cgpa >= 0.0,
            self.cgpa <= 10.0,
            self.year >= 1,
            self.year <= 4,
        ])


class RuleCompiler:
    """Compiles JSON policy rules and student facts into a Z3PolicyContext."""

    def __init__(self, policies_dict: Dict[str, Any]):
        self.courses: Dict[str, CourseDefinition] = {
            c["code"]: CourseDefinition.model_validate(c) for c in policies_dict["courses"]
        }
        self.rules: Dict[str, AcademicRule] = {
            r["id"]: AcademicRule.model_validate(r) for r in policies_dict["rules"]
        }

    def compile(self, student: StudentProfile, target_course: str) -> Z3PolicyContext:
        ctx = Z3PolicyContext(target_course, self.courses)

        # 1. Compile student known/unknown facts into structural constraints
        fact_constraints, unknown_fields = self._compile_student_facts(student, ctx)
        ctx.structural_constraints.extend(fact_constraints)
        ctx.unknown_fields = unknown_fields

        # 2. Compile applicable rules for target course
        for rule_id, rule in self.rules.items():
            if not self._is_rule_applicable(rule, target_course):
                continue

            expr = self._compile_rule_expression(rule, ctx, student)
            if expr is not None:
                label_var = z3.Bool(f"rule_active_{rule_id}")
                ctx.rule_assertions[rule_id] = (expr, label_var)

        return ctx

    def _is_rule_applicable(self, rule: AcademicRule, target_course: str) -> bool:
        if rule.target_courses:
            return target_course in rule.target_courses
        return True

    def _compile_rule_expression(
        self, rule: AcademicRule, ctx: Z3PolicyContext, student: StudentProfile
    ) -> Optional[z3.BoolRef]:
        # Prerequisite rules
        if rule.type == "prerequisite":
            if not rule.prerequisites:
                return None  # No prerequisite required (R001)

            prereq_checks = []
            for item in rule.prerequisites:
                p_code = item.course_code
                p_completed = ctx.get_completed_var(p_code)

                if item.waiver_allowed:
                    p_waiver = ctx.get_waiver_var(ctx.target_course, p_code)
                    prereq_checks.append(z3.Or(p_completed, p_waiver))
                else:
                    prereq_checks.append(p_completed)

            return z3.And(*prereq_checks) if prereq_checks else None

        # Minimum CGPA rules
        if rule.type == "minimum_cgpa" and rule.condition:
            return ctx.cgpa >= float(rule.condition.value)

        # Probation restriction
        if rule.type == "probation_restriction":
            return ctx.on_probation == False

        # Standing year rules
        if rule.type == "year_standing" and rule.condition:
            return ctx.year >= int(rule.condition.value)

        # Credit limits and overload exception (Rules R012, R013, R014)
        if rule.id == "R012":
            # Normal credit limit applies UNLESS final year exception (R013) is satisfied
            extra_allowed = z3.And(ctx.year == 4, ctx.cgpa >= 8.0, ctx.department_credit_approval == True)
            return z3.If(extra_allowed, ctx.total_credits <= 28, ctx.total_credits <= 24)

        if rule.id == "R013":
            # Department approval exception clause
            return z3.Implies(
                ctx.total_credits > 24,
                z3.And(ctx.year == 4, ctx.cgpa >= 8.0, ctx.department_credit_approval == True),
            )

        if rule.id == "R014":
            # Absolute ceiling of 28 credits
            return ctx.total_credits <= 28

        # Course repetition restriction (R019)
        if rule.type == "repeat_restriction":
            return ctx.course_already_completed == False

        # Course offered restriction (R020)
        if rule.type == "course_offered":
            return ctx.course_offered == True

        # Admin hold restriction (R023)
        if rule.type == "admin_hold":
            return ctx.registration_hold == False

        return None

    def _compile_student_facts(
        self, student: StudentProfile, ctx: Z3PolicyContext
    ) -> Tuple[List[z3.BoolRef], List[str]]:
        facts: List[z3.BoolRef] = []
        unknowns: List[str] = []

        # CGPA
        if student.cgpa is not None:
            facts.append(ctx.cgpa == float(student.cgpa))
        else:
            unknowns.append("cgpa")

        # Academic Year
        if student.year is not None:
            facts.append(ctx.year == int(student.year))
        else:
            unknowns.append("year")

        # Current registered credits
        if student.current_registered_credits is not None:
            facts.append(ctx.credits_current == int(student.current_registered_credits))
        else:
            unknowns.append("current_registered_credits")

        # Probation
        if student.on_probation is not None:
            facts.append(ctx.on_probation == bool(student.on_probation))
        else:
            unknowns.append("on_probation")

        # Registration Hold
        if student.registration_hold is not None:
            facts.append(ctx.registration_hold == bool(student.registration_hold))
        else:
            unknowns.append("registration_hold")

        # Department credit overload approval
        if student.department_credit_approval is not None:
            facts.append(ctx.department_credit_approval == bool(student.department_credit_approval))
        else:
            unknowns.append("department_credit_approval")

        # Course completion facts
        if student.completed_courses is not None:
            completed_set = set(student.completed_courses)
            for c_code in ctx.catalog.keys():
                is_completed = c_code in completed_set
                facts.append(ctx.get_completed_var(c_code) == is_completed)
            facts.append(ctx.course_already_completed == (ctx.target_course in completed_set))
        else:
            unknowns.append("completed_courses")

        # Course offered status (from catalog)
        target_info = ctx.catalog.get(ctx.target_course)
        is_offered = target_info.is_offered if target_info else True
        facts.append(ctx.course_offered == is_offered)

        # Faculty waivers
        if student.faculty_waivers is not None:
            for p_code in ctx.catalog.keys():
                has_w = student.has_waiver_for(ctx.target_course, p_code)
                facts.append(ctx.get_waiver_var(ctx.target_course, p_code) == has_w)
        else:
            unknowns.append("faculty_waivers")

        return facts, unknowns
