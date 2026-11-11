"""
Formal Reasoning Engine powered by Z3.
Implements:
- 3-Valued Eligibility Logic (ELIGIBLE, INELIGIBLE, UNDETERMINED, CONFLICT)
- Conflict Detection via UNSAT core
- Counterexample Generation
- Missing Information Identification
- Grounded Natural Language Explanations
"""

from typing import Dict, List, Optional, Any
import json
from pathlib import Path
from pydantic import BaseModel, Field
import z3

from reasoning.rule_schema import (
    StudentProfile,
    AcademicRule,
    DecisionEnum,
    SourceReference,
)
from reasoning.rule_compiler import RuleCompiler, Z3PolicyContext


class EligibilityResult(BaseModel):
    decision: DecisionEnum
    target_course: str
    student_id: str
    failing_rules: List[str] = Field(default_factory=list)
    failing_rule_descriptions: List[str] = Field(default_factory=list)
    missing_facts: List[str] = Field(default_factory=list)
    conflicting_rules: List[str] = Field(default_factory=list)
    counterexample: Optional[Dict[str, Any]] = None
    explanation: str
    source_references: List[SourceReference] = Field(default_factory=list)


class Z3Reasoner:
    """Z3 Formal Reasoning Engine for Academic Policies."""

    def __init__(self, policies_path: Optional[Path] = None):
        if policies_path is None:
            policies_path = Path(__file__).parent.parent / "data" / "policies.json"

        with open(policies_path, "r", encoding="utf-8") as f:
            self.policies_dict = json.load(f)

        self.compiler = RuleCompiler(self.policies_dict)
        self.rules_lookup: Dict[str, AcademicRule] = {
            r["id"]: AcademicRule.model_validate(r) for r in self.policies_dict["rules"]
        }

    def evaluate(self, student: StudentProfile, target_course: str) -> EligibilityResult:
        ctx = self.compiler.compile(student, target_course)

        # 1. Fact Consistency Check: Are the student's constraints internally valid?
        fact_solver = z3.Solver()
        for sc in ctx.structural_constraints:
            fact_solver.add(sc)

        if fact_solver.check() == z3.unsat:
            return EligibilityResult(
                decision=DecisionEnum.CONFLICT,
                target_course=target_course,
                student_id=student.student_id,
                explanation=f"Contradictory student facts detected for {student.student_id}.",
                source_references=[],
            )

        # 2. Ineligibility Check via UNSAT Core
        # If student's known facts + applicable rules is UNSAT, student cannot be eligible.
        inelig_solver = z3.Solver()
        inelig_solver.set(unsat_core=True)

        for sc in ctx.structural_constraints:
            inelig_solver.add(sc)

        for rule_id, (rule_expr, label_var) in ctx.rule_assertions.items():
            inelig_solver.assert_and_track(rule_expr, label_var)

        if inelig_solver.check() == z3.unsat:
            unsat_core = inelig_solver.unsat_core()
            core_rule_ids = {
                str(lbl).replace("rule_active_", "")
                for lbl in unsat_core
                if str(lbl).startswith("rule_active_")
            }

            # Find all definitively violated rules (so the explanation covers all blockers)
            violated_rule_ids = set(self._identify_all_violated_rules(ctx))
            failing_rule_ids = sorted(list(core_rule_ids.union(violated_rule_ids)))

            descriptions = [
                f"{rid}: {self.rules_lookup[rid].description}"
                for rid in failing_rule_ids
                if rid in self.rules_lookup
            ]
            sources = [
                self.rules_lookup[rid].source for rid in failing_rule_ids if rid in self.rules_lookup
            ]

            explanation = (
                f"Student {student.student_id} is INELIGIBLE for {target_course}. "
                f"The request violates the following policies: " + "; ".join(descriptions)
            )

            return EligibilityResult(
                decision=DecisionEnum.INELIGIBLE,
                target_course=target_course,
                student_id=student.student_id,
                failing_rules=failing_rule_ids,
                failing_rule_descriptions=descriptions,
                explanation=explanation,
                source_references=sources,
            )

        # 3. Guaranteed Eligibility Check
        # Check if there exists ANY scenario where ANY applicable rule is violated.
        elig_solver = z3.Solver()
        for sc in ctx.structural_constraints:
            elig_solver.add(sc)

        violations = [z3.Not(rule_expr) for _, (rule_expr, _) in ctx.rule_assertions.items()]
        if violations:
            elig_solver.add(z3.Or(*violations))

        if not violations or elig_solver.check() == z3.unsat:
            # No rule can be violated in any possible assignment -> ELIGIBLE
            applied_rule_ids = list(ctx.rule_assertions.keys())
            sources = [
                self.rules_lookup[rid].source for rid in applied_rule_ids if rid in self.rules_lookup
            ]
            return EligibilityResult(
                decision=DecisionEnum.ELIGIBLE,
                target_course=target_course,
                student_id=student.student_id,
                explanation=f"Student {student.student_id} is fully ELIGIBLE for {target_course}. All academic prerequisites, GPA requirements, and credit limits are satisfied.",
                source_references=sources,
            )

        # 4. UNDETERMINED: Both Eligible and Ineligible remain possible due to missing facts
        missing = list(ctx.unknown_fields)
        counterexample = self._extract_counterexample_model(elig_solver, ctx)

        sources = [
            self.rules_lookup[rid].source
            for rid in ctx.rule_assertions.keys()
            if rid in self.rules_lookup
        ]

        explanation = (
            f"Registration eligibility for {target_course} is UNDETERMINED. "
            f"Required student information is missing: {', '.join(missing)}. "
            f"Depending on these missing attributes, registration could be either approved or denied."
        )

        return EligibilityResult(
            decision=DecisionEnum.UNDETERMINED,
            target_course=target_course,
            student_id=student.student_id,
            missing_facts=missing,
            counterexample=counterexample,
            explanation=explanation,
            source_references=sources,
        )

    def _identify_all_violated_rules(self, ctx: Z3PolicyContext) -> List[str]:
        violated = []
        for rid, (expr, _) in ctx.rule_assertions.items():
            s = z3.Solver()
            for sc in ctx.structural_constraints:
                s.add(sc)
            s.add(expr)
            if s.check() == z3.unsat:
                violated.append(rid)
        return violated

    def _extract_counterexample_model(
        self, solver: z3.Solver, ctx: Z3PolicyContext
    ) -> Optional[Dict[str, Any]]:
        """Extracts variable assignments from a satisfying model."""
        if solver.check() == z3.sat:
            m = solver.model()
            result = {}
            for d in m.decls():
                name = d.name()
                if not name.startswith("rule_active_"):
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
        return None
