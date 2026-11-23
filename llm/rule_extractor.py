"""
Rule Extractor: Transforms natural language policy passages into Pydantic AcademicRule models using LLMs.
"""

import json
import re
from typing import Optional, Dict, Any
from ingestion.source_tracker import PolicyPassage
from reasoning.rule_schema import AcademicRule
from llm.client import LLMClient
from llm.prompts import RULE_EXTRACTION_SYSTEM_PROMPT


class LLMRuleExtractor:
    """Extracts formal rules from policy text with Pydantic schema validation."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def extract_rule_from_passage(self, passage: PolicyPassage) -> Optional[AcademicRule]:
        """Prompts LLM to extract a single AcademicRule from a passage."""
        prompt = (
            f"Extract the formal academic rule from this passage:\n"
            f"Passage ID: {passage.passage_id}\n"
            f"Section: {passage.section}\n"
            f"Document: {passage.document}\n"
            f"Content: \"{passage.text}\"\n"
        )

        raw_output = self.llm.generate(prompt, system_prompt=RULE_EXTRACTION_SYSTEM_PROMPT)
        cleaned_json = self._clean_json(raw_output)

        try:
            data = json.loads(cleaned_json)
            # Ensure source links back properly
            if "source" not in data or not isinstance(data["source"], dict):
                data["source"] = {
                    "document": passage.document,
                    "section": passage.section,
                    "passage_id": passage.passage_id,
                    "text": passage.text,
                }
            rule = AcademicRule.model_validate(data)
            return rule
        except Exception as e:
            print(f"Failed to parse rule for {passage.passage_id}: {e}")
            return None

    def _clean_json(self, raw_str: str) -> str:
        """Removes markdown code fences and extraneous text."""
        s = raw_str.strip()
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()
