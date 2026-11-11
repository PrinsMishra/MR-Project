"""
Contradictory Policy and Inconsistency Detector using Z3 UNSAT Cores.

Detects:
1. Universal Contradictions: Rules that make eligibility mathematically impossible for any student.
2. Rule Pairwise / Conditional Inconsistencies: Situations where two rules enforce contradictory requirements.
3. Candidate Rule Ingestion Conflicts: Verifies whether a newly extracted LLM rule conflicts with existing approved rules.
"""

from typing import Dict, List, Optional, Any, NamedTuple
import json
from pathlib import Path
import z3

from reasoning.rule_schema import AcademicRule, CourseDefinition
from reasoning.rule_compiler import Z3PolicyContext


class ConflictReport(NamedTuple):
    has_conflict: bool
    target_course: str
    conflicting_rule_ids: List[str]
    conflicting_rule_descriptions: List[str]
    explanation: str
    unsat_core_labels: List[str]


class ConflictDetector:
    """Uses Z3 labelled assertions and unsat cores to pinpoint contradictory academic policies."""

    def __init__(self, policies_path: Optional[Path] = None):
        if policies_path is None:
            policies_path = Path(__file__).parent.parent / "data" / "policies.json"

        with open(policies_path, "r", encoding="utf-8") as f:
            self.policies_dict = json.load(f)

        self.courses = {
            c["code"]: CourseDefinition.model_validate(c) for c in self.policies_dict["courses"]
        }
        self.rules = {
            r["id"]: AcademicRule.model_validate(r) for r in self.policies_dict["rules"]
        }

    def check_policy_set_consistency(
        self, target_course: str, custom_rules: Optional[List[AcademicRule]] = None
    ) -> ConflictReport:
        """
        Tests whether the conjunction of all applicable policies is satisfiable
        for at least one theoretical student in the domain.
        If UNSAT, extracts the minimal conflicting subset via unsat_core().
        """
        rules_to_check = custom_rules if custom_rules is not None else list(self.rules.values())

        ctx = Z3PolicyContext(target_course, self.courses)
        solver = z3.Solver()
        solver.set(unsat_core=True)

        for sc in ctx.structural_constraints:
            solver.add(sc)

        label_map: Dict[str, AcademicRule] = {}

        for rule in rules_to_check:
            if rule.target_courses and target_course not in rule.target_courses:
                continue

            expr = self._compile_rule(rule, ctx)
            if expr is not None:
                lbl = z3.Bool(f"conflict_track_{rule.id}")
                label_map[str(lbl)] = rule
                solver.assert_and_track(expr, lbl)

        if solver.check() == z3.unsat:
            core = solver.unsat_core()
            conflicting_ids = [label_map[str(lbl)].id for lbl in core if str(lbl) in label_map]
            conflicting_descs = [
                f"{label_map[str(lbl)].id}: {label_map[str(lbl)].description}"
                for lbl in core
                if str(lbl) in label_map
            ]

            explanation = (
                f"Contradictory policy detected for course {target_course}! "
                f"No student can ever satisfy the following rules simultaneously: "
                + "; ".join(conflicting_descs)
            )

            return ConflictReport(
                has_conflict=True,
                target_course=target_course,
                conflicting_rule_ids=conflicting_ids,
                conflicting_rule_descriptions=conflicting_descs,
                explanation=explanation,
                unsat_core_labels=[str(lbl) for lbl in core],
            )

        return ConflictReport(
            has_conflict=False,
            target_course=target_course,
            conflicting_rule_ids=[],
            conflicting_rule_descriptions=[],
            explanation=f"Policies for {target_course} are internally consistent and satisfiable.",
            unsat_core_labels=[],
        )

    def test_candidate_rule_conflict(
        self, candidate_rule: AcademicRule, target_course: str
    ) -> ConflictReport:
        """
        Simulates ingesting a newly extracted candidate rule from an LLM.
        Checks if adding this candidate rule causes an UNSAT conflict with existing approved rules.
        """
        existing_rules = list(self.rules.values())
        combined = existing_rules + [candidate_rule]
        return self.check_policy_set_consistency(target_course, custom_rules=combined)

    def _compile_rule(self, rule: AcademicRule, ctx: Z3PolicyContext) -> Optional[z3.BoolRef]:
        # If the rule specifies an explicit logical condition (atomic or compound), compile directly
        if rule.condition:
            return self._compile_condition(rule.condition, ctx)

        if rule.type == "prerequisite":
            if not rule.prerequisites:
                return None
            checks = []
            for item in rule.prerequisites:
                p_var = ctx.get_completed_var(item.course_code)
                if item.waiver_allowed:
                    w_var = ctx.get_waiver_var(ctx.target_course, item.course_code)
                    checks.append(z3.Or(p_var, w_var))
                else:
                    checks.append(p_var)
            return z3.And(*checks)

        if rule.type == "probation_restriction":
            return ctx.on_probation == False

        if rule.id == "R012":
            extra = z3.And(ctx.year == 4, ctx.cgpa >= 8.0, ctx.department_credit_approval == True)
            return z3.If(extra, ctx.total_credits <= 28, ctx.total_credits <= 24)

        if rule.id == "R014":
            return ctx.total_credits <= 28

        if rule.type == "repeat_restriction":
            return ctx.course_already_completed == False

        if rule.type == "course_offered":
            return ctx.course_offered == True

        if rule.type == "admin_hold":
            return ctx.registration_hold == False

        return None

    def _compile_condition(self, cond: Any, ctx: Z3PolicyContext) -> Optional[z3.BoolRef]:
        if hasattr(cond, "logical_op") and hasattr(cond, "conditions"):
            # CompoundCondition
            sub_exprs = [self._compile_condition(c, ctx) for c in cond.conditions]
            sub_exprs = [e for e in sub_exprs if e is not None]
            if not sub_exprs:
                return None
            if cond.logical_op == "AND":
                return z3.And(*sub_exprs)
            if cond.logical_op == "OR":
                return z3.Or(*sub_exprs)
            if cond.logical_op == "NOT":
                return z3.Not(sub_exprs[0])

        if hasattr(cond, "field") and hasattr(cond, "operator"):
            # AtomicCondition
            f = cond.field
            v = cond.value
            op = cond.operator
            if f == "cgpa":
                if op == ">=": return ctx.cgpa >= float(v)
                if op == "<=": return ctx.cgpa <= float(v)
                if op == "<": return ctx.cgpa < float(v)
                if op == ">": return ctx.cgpa > float(v)
                if op == "==": return ctx.cgpa == float(v)
            if f == "total_credits":
                if op == ">=": return ctx.total_credits >= int(v)
                if op == "<=": return ctx.total_credits <= int(v)
                if op == "<": return ctx.total_credits < int(v)
                if op == ">": return ctx.total_credits > int(v)
                if op == "==": return ctx.total_credits == int(v)
            if f == "year":
                if op == ">=": return ctx.year >= int(v)
                if op == "<=": return ctx.year <= int(v)
                if op == "==": return ctx.year == int(v)
            if f == "on_probation":
                return ctx.on_probation == bool(v)
            if f == "registration_hold":
                return ctx.registration_hold == bool(v)

        return None
