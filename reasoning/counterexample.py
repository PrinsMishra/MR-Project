"""
Counterexample Generator for Academic Policy Verification.

Given an eligibility claim or hypothesis (e.g., from an LLM or user),
searches for a concrete student assignment that satisfies all actual policies
while falsifying the claim (or proving the claim invalid).
"""

from typing import Dict, List, Optional, Any, NamedTuple
import json
from pathlib import Path
import z3

from reasoning.rule_schema import AcademicRule, CourseDefinition
from reasoning.rule_compiler import Z3PolicyContext


class CounterexampleResult(NamedTuple):
    found: bool
    hypothesis: str
    counterexample: Optional[Dict[str, Any]]
    explanation: str
    violated_rule_id: Optional[str] = None


class CounterexampleGenerator:
    """
    Formal counterexample search using Z3.
    Formulates: Actual_Policies ∧ ¬Hypothesis (or Hypothesis ∧ ¬Actual_Policies)
    """

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

    def find_counterexample_for_claim(
        self,
        target_course: str,
        hypothesis_premise: z3.BoolRef,
        hypothesis_claim: z3.BoolRef,
        hypothesis_description: str,
    ) -> CounterexampleResult:
        """
        Tests whether Premise implies Claim under the official university policies.
        Specifically, checks if there exists a student satisfying:
            Actual_Policies ∧ Premise ∧ ¬Claim
        If SAT, the satisfying assignment is a concrete COUNTEREXAMPLE to the claim.
        """
        ctx = Z3PolicyContext(target_course, self.courses)

        solver = z3.Solver()
        for sc in ctx.structural_constraints:
            solver.add(sc)

        # Build true eligibility condition from applicable rules
        rule_conditions = []
        rule_map = {}
        for rid, rule in self.rules.items():
            if rule.target_courses and target_course not in rule.target_courses:
                continue

            expr = self._compile_rule(rule, ctx)
            if expr is not None:
                rule_conditions.append(expr)
                rule_map[expr] = rid

        true_eligibility = z3.And(*rule_conditions) if rule_conditions else z3.BoolVal(True)

        # We want to falsify the hypothesis:
        # Hypothesis asserts: Premise => Claim
        # Counterexample exists if: Premise is True, but True_Eligibility != Claim (or True_Eligibility violates Claim)
        # Specifically, if claim is "student is eligible", counterexample is: Premise holds, student is INELIGIBLE
        solver.add(hypothesis_premise)
        solver.add(hypothesis_claim != true_eligibility)

        if solver.check() == z3.sat:
            m = solver.model()
            student_assignment = self._model_to_dict(m, ctx)

            # Identify which actual rule was violated in the counterexample
            violated_rid = None
            for expr, rid in rule_map.items():
                if is_falsified(m, expr):
                    violated_rid = rid
                    break

            explanation = (
                f"Counterexample found for claim: '{hypothesis_description}'. "
                f"Under the actual university policies, a student with the generated profile "
                f"would have eligibility={m.eval(true_eligibility)}, violating the claim."
            )
            if violated_rid:
                explanation += f" Specifically, rule {violated_rid} ({self.rules[violated_rid].title}) is violated."

            return CounterexampleResult(
                found=True,
                hypothesis=hypothesis_description,
                counterexample=student_assignment,
                explanation=explanation,
                violated_rule_id=violated_rid,
            )

        return CounterexampleResult(
            found=False,
            hypothesis=hypothesis_description,
            counterexample=None,
            explanation=f"No counterexample exists. The claim '{hypothesis_description}' holds under all valid university policies.",
            violated_rule_id=None,
        )

    # ---------------- Pre-packaged Academic Claim Checkers ----------------

    def check_claim_high_gpa_implies_eligible(self, target_course: str, min_gpa: float = 8.0) -> CounterexampleResult:
        """
        Tests the naive LLM claim: 'All students with CGPA >= min_gpa are eligible for target_course.'
        """
        ctx = Z3PolicyContext(target_course, self.courses)
        premise = ctx.cgpa >= min_gpa
        claim = z3.BoolVal(True)  # Claim: student is eligible
        desc = f"All students with CGPA >= {min_gpa} are eligible for {target_course}"
        return self.find_counterexample_for_claim(target_course, premise, claim, desc)

    def check_claim_prerequisites_alone_imply_eligible(self, target_course: str) -> CounterexampleResult:
        """
        Tests the claim: 'Students who completed all course prerequisites are always eligible.'
        """
        ctx = Z3PolicyContext(target_course, self.courses)
        # Prereqs for CS301 are CS101 and CS201
        premise = z3.And(ctx.get_completed_var("CS101") == True, ctx.get_completed_var("CS201") == True)
        claim = z3.BoolVal(True)
        desc = f"Any student who completed CS101 and CS201 is eligible for {target_course}"
        return self.find_counterexample_for_claim(target_course, premise, claim, desc)

    def check_claim_approval_alone_grants_28_credits(self, target_course: str) -> CounterexampleResult:
        """
        Tests the claim: 'Any student with department approval can register for 28 credits.'
        (Counterexample: student with dept approval but year != 4 or CGPA < 8.0).
        """
        ctx = Z3PolicyContext(target_course, self.courses)
        premise = z3.And(ctx.department_credit_approval == True, ctx.total_credits == 28)
        claim = z3.BoolVal(True)
        desc = f"Any student with department approval can register for 28 credits in {target_course}"
        return self.find_counterexample_for_claim(target_course, premise, claim, desc)

    def check_claim_waiver_for_other_course_applies(
        self, target_course: str, waived_course: str, prereq_code: str
    ) -> CounterexampleResult:
        """
        Tests the claim: 'A waiver granted for another course waives the prerequisite for this course.'
        (Verifies Rule R021: waiver isolation).
        """
        ctx = Z3PolicyContext(target_course, self.courses)
        premise = z3.And(
            ctx.get_waiver_var(waived_course, prereq_code) == True,
            ctx.get_completed_var(prereq_code) == False,
        )
        claim = z3.BoolVal(True)
        desc = f"A waiver for {waived_course}/{prereq_code} also satisfies prerequisites for {target_course}"
        return self.find_counterexample_for_claim(target_course, premise, claim, desc)

    def check_claim_good_standing_implies_elective_eligible(self, target_course: str) -> CounterexampleResult:
        """
        Tests the claim: 'Students with CGPA >= 8.0 and not on probation can take 400-level courses.'
        (Counterexample: Year 1 or Year 2 students, violating R017/R018).
        """
        ctx = Z3PolicyContext(target_course, self.courses)
        premise = z3.And(ctx.cgpa >= 8.0, ctx.on_probation == False)
        claim = z3.BoolVal(True)
        desc = f"Any student not on probation with CGPA >= 8.0 is eligible for {target_course}"
        return self.find_counterexample_for_claim(target_course, premise, claim, desc)

    # ---------------- Helpers ----------------

    def _compile_rule(self, rule: AcademicRule, ctx: Z3PolicyContext) -> Optional[z3.BoolRef]:
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

        if rule.type == "minimum_cgpa" and rule.condition:
            return ctx.cgpa >= float(rule.condition.value)

        if rule.type == "probation_restriction":
            return ctx.on_probation == False

        if rule.type == "year_standing" and rule.condition:
            return ctx.year >= int(rule.condition.value)

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

    def _model_to_dict(self, m: z3.ModelRef, ctx: Z3PolicyContext) -> Dict[str, Any]:
        result = {}
        for d in m.decls():
            name = d.name()
            val = m[d]
            if z3.is_real(val):
                result[name] = float(val.as_fraction())
            elif z3.is_int(val):
                result[name] = val.as_long()
            elif z3.is_bool(val):
                result[name] = bool(val)
            else:
                result[name] = str(val)
        return result


def is_falsified(m: z3.ModelRef, expr: z3.BoolRef) -> bool:
    try:
        val = m.eval(expr)
        return z3.is_false(val)
    except Exception:
        return False
