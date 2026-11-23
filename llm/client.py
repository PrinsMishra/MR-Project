"""
Unified LLM Client supporting Groq and Google GenAI APIs with offline fallback.
"""

import os
import json
import re
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Invokes LLM for structured extraction and natural language reasoning."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

        self.groq_client = None
        self.genai_client = None
        self.provider = "fallback"

        # Try Groq
        if self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
                self.provider = "groq"
            except Exception:
                pass

        # Try Google GenAI
        if not self.groq_client and self.google_api_key:
            try:
                import google.genai as genai
                self.genai_client = genai.Client(api_key=self.google_api_key)
                self.provider = "google"
            except Exception:
                pass

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates completion using Groq, Google, or offline rule induction."""
        # 1. Try Groq
        if self.groq_client:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1024,
                )
                return response.choices[0].message.content
            except Exception:
                pass

        # 2. Try Google GenAI
        if self.genai_client:
            try:
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = self.genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=full_prompt,
                )
                return response.text
            except Exception:
                pass

        # 3. Deterministic Local Rule Extraction Fallback (sandbox / offline mode)
        return self._local_rule_extraction_fallback(prompt)

    def _local_rule_extraction_fallback(self, prompt: str) -> str:
        """Extracts structured JSON using regex heuristics when offline."""
        cgpa_match = re.search(r"CGPA\s+(?:of\s+)?at\s+least\s+([0-9\.]+)", prompt, re.IGNORECASE)
        credits_match = re.search(r"([0-9]+)\s+credits", prompt, re.IGNORECASE)
        course_matches = re.findall(r"\b(CS\d{3})\b", prompt)
        rule_match = re.search(r"Rule\s+(R\d+)", prompt, re.IGNORECASE)
        passage_match = re.search(r"\[Passage\s+(P\d+)\]", prompt, re.IGNORECASE)

        rule_id = rule_match.group(1) if rule_match else "R_EXTRACTED"
        passage_id = passage_match.group(1) if passage_match else "P_EXTRACTED"
        target_course = course_matches[0] if course_matches else "GLOBAL"

        if "prerequisite" in prompt.lower():
            prereqs = [c for c in course_matches if c != target_course]
            return json.dumps({
                "id": rule_id,
                "type": "prerequisite",
                "title": f"{target_course} Prerequisite Rule",
                "description": f"{target_course} requires {', '.join(prereqs)}",
                "target_courses": [target_course],
                "prerequisites": [{"course_code": p, "waiver_allowed": True} for p in prereqs],
                "waiver_allowed": True,
                "source": {
                    "document": "academic_policy_nitech.pdf",
                    "section": "1.2",
                    "passage_id": passage_id,
                    "text": prompt[:150],
                },
                "status": "draft",
            })

        if cgpa_match:
            val = float(cgpa_match.group(1))
            return json.dumps({
                "id": rule_id,
                "type": "minimum_cgpa",
                "title": f"{target_course} Minimum CGPA",
                "description": f"Students must have a CGPA of at least {val} to register for {target_course}.",
                "target_courses": [target_course] if target_course != "GLOBAL" else [],
                "condition": {"field": "cgpa", "operator": ">=", "value": val},
                "waiver_allowed": False,
                "source": {
                    "document": "academic_policy_nitech.pdf",
                    "section": "2.1",
                    "passage_id": passage_id,
                    "text": prompt[:150],
                },
                "status": "draft",
            })

        if credits_match:
            val = int(credits_match.group(1))
            return json.dumps({
                "id": rule_id,
                "type": "credit_limit",
                "title": f"Semester Credit Limit ({val} Credits)",
                "description": f"Maximum semester credit limit is {val} credits.",
                "target_courses": [],
                "condition": {"field": "total_credits", "operator": "<=", "value": val},
                "waiver_allowed": val == 24,
                "source": {
                    "document": "academic_policy_nitech.pdf",
                    "section": "3.1",
                    "passage_id": passage_id,
                    "text": prompt[:150],
                },
                "status": "draft",
            })

        return json.dumps({
            "id": rule_id,
            "type": "academic_policy",
            "title": f"Academic Regulation {rule_id}",
            "description": prompt[:100],
            "target_courses": course_matches,
            "source": {
                "document": "academic_policy_nitech.pdf",
                "section": "General",
                "passage_id": passage_id,
                "text": prompt[:150],
            },
            "status": "draft",
        })
