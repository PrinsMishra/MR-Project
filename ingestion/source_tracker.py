"""
Source Tracking and Metadata Management for Ingested Academic Policies.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PolicyPassage(BaseModel):
    passage_id: str = Field(..., description="Unique passage identifier, e.g. 'P001'")
    document: str = Field(..., description="Source document file name or reference")
    page_number: int = Field(1, description="Page number where the passage appears (1-based)")
    section: str = Field("", description="Section identifier, e.g. '1.2' or '2.1'")
    text: str = Field(..., description="Full text content of the passage")
    rule_hints: List[str] = Field(default_factory=list, description="Rule IDs referenced in passage")
    course_hints: List[str] = Field(default_factory=list, description="Course codes referenced in passage")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional provenance")


class SourceTracker:
    """Maintains a persistent index of all ingested passages across documents."""

    def __init__(self):
        self.passages: Dict[str, PolicyPassage] = {}

    def register_passage(self, passage: PolicyPassage):
        self.passages[passage.passage_id] = passage

    def get_passage(self, passage_id: str) -> Optional[PolicyPassage]:
        return self.passages.get(passage_id)

    def find_by_course(self, course_code: str) -> List[PolicyPassage]:
        return [
            p for p in self.passages.values()
            if course_code in p.course_hints
        ]

    def find_by_rule(self, rule_id: str) -> Optional[PolicyPassage]:
        for p in self.passages.values():
            if rule_id in p.rule_hints:
                return p
        return None
