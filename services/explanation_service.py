"""
Explanation Service: Synthesizes explainable, verifiable markdown reports
grounded in Z3 proofs and RAG retrieved passages.
"""

from services.eligibility_service import VerificationBundle
from reasoning.rule_schema import DecisionEnum


class ExplanationService:
    """Renders formatted explanations and breakdown cards for web and chat interfaces."""

    @staticmethod
    def render_markdown_summary(bundle: VerificationBundle) -> str:
        lines = []

        # Decision Header
        badge_map = {
            DecisionEnum.ELIGIBLE: "🟢 **STATUS: ELIGIBLE**",
            DecisionEnum.INELIGIBLE: "🔴 **STATUS: INELIGIBLE (REGISTRATION BLOCKED)**",
            DecisionEnum.UNDETERMINED: "🟠 **STATUS: UNDETERMINED (INCOMPLETE INFORMATION)**",
            DecisionEnum.CONFLICT: "🟣 **STATUS: POLICY CONFLICT DETECTED**",
        }
        lines.append(f"### {badge_map.get(bundle.decision, bundle.decision.value)}")
        lines.append(f"**Student:** {bundle.student_name} (`{bundle.student_id}`) | **Target Course:** `{bundle.target_course}`\n")
        lines.append(f"> {bundle.explanation}\n")

        # Ineligible details with citations
        if bundle.decision == DecisionEnum.INELIGIBLE and bundle.failing_rule_details:
            lines.append("#### ❌ Violated Policies (Identified via Z3 UNSAT Core)")
            for r in bundle.failing_rule_details:
                lines.append(
                    f"* **[{r['rule_id']}] {r['title']}**: {r['description']}\n"
                    f"  * *Official Source:* `{r['source_doc']}` (Section {r['source_section']}, Passage `{r['passage_id']}`)\n"
                    f"  * *Verbatim Text:* *\"{r['source_text']}\"*\n"
                )

        # Undetermined details with missing facts
        if bundle.decision == DecisionEnum.UNDETERMINED:
            lines.append("#### ⚠️ Required Missing Information")
            lines.append("The formal solver determined that registration is undecided because the following attributes are missing:\n")
            for f in bundle.missing_facts:
                lines.append(f"* **`{f}`**: Needs to be provided by the student or registrar to reach a definitive decision.")

        # Counterexample model
        if bundle.counterexample_model:
            lines.append("\n#### 🔬 Concrete Counterexample Model (Z3 Satisfying Assignment)")
            lines.append("The solver identified a concrete assignment of variables where registration would be refused:")
            lines.append("```json")
            import json
            lines.append(json.dumps(bundle.counterexample_model, indent=2))
            lines.append("```")

        # RAG Retrieved passages
        if bundle.retrieved_passages:
            lines.append("\n#### 📚 Supporting RAG Policy Passages")
            for p in bundle.retrieved_passages[:3]:
                lines.append(f"* **[{p.passage_id}] {p.document} §{p.section}** (Relevance Score: `{p.score:.3f}`):")
                lines.append(f"  > *\"{p.text}\"*")

        return "\n".join(lines)
