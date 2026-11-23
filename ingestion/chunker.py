"""
Document Chunker with Passage and Source Tracking.
Splits text into overlapping or passage-aligned chunks tagged with sections and course hints.
"""

import re
from typing import List, Dict, Any
from pathlib import Path
from ingestion.pdf_parser import PDFParser
from ingestion.source_tracker import PolicyPassage


class PolicyChunker:
    """
    Chunks academic policies while preserving rule and section boundaries.
    Detects passage tags like [Passage P001] or sections like SECTION 1.
    """

    COURSE_PATTERN = re.compile(r"\b(CS\d{3})\b")
    RULE_PATTERN = re.compile(r"\b(R\d{3})\b")
    PASSAGE_PATTERN = re.compile(r"\[Passage\s+(P\d+)\]\s*(Rule\s+R\d+:\s*)?", re.IGNORECASE)
    SECTION_PATTERN = re.compile(r"SECTION\s+(\d+(?:\.\d+)?)", re.IGNORECASE)

    def chunk_pdf(self, pdf_path: Path) -> List[PolicyPassage]:
        parser = PDFParser(pdf_path)
        pages = parser.extract_pages()
        passages: List[PolicyPassage] = []

        current_section = "General"

        for page_data in pages:
            page_num = page_data["page_number"]
            doc_name = page_data["document"]
            text = page_data["text"]

            lines = text.split("\n")
            current_passage_id = None
            current_passage_lines = []

            for line in lines:
                clean = line.strip()
                if not clean:
                    continue

                sec_match = self.SECTION_PATTERN.search(clean)
                if sec_match:
                    current_section = sec_match.group(1)

                p_match = self.PASSAGE_PATTERN.search(clean)
                if p_match:
                    # Flush previous passage if any
                    if current_passage_id and current_passage_lines:
                        p_text = " ".join(current_passage_lines)
                        passages.append(self._create_passage(
                            current_passage_id, doc_name, page_num, current_section, p_text
                        ))
                    current_passage_id = p_match.group(1)
                    current_passage_lines = [clean]
                else:
                    if current_passage_id:
                        current_passage_lines.append(clean)
                    elif clean.startswith("Rule R") or clean.startswith("CS"):
                        # Unflagged rule line
                        current_passage_lines.append(clean)

            # Flush last passage on page
            if current_passage_id and current_passage_lines:
                p_text = " ".join(current_passage_lines)
                passages.append(self._create_passage(
                    current_passage_id, doc_name, page_num, current_section, p_text
                ))

        # Fallback if no passage tags found: chunk by paragraph
        if not passages:
            passages = self._fallback_paragraph_chunking(pages)

        return passages

    def _create_passage(
        self, passage_id: str, doc_name: str, page_num: int, section: str, text: str
    ) -> PolicyPassage:
        courses = list(set(self.COURSE_PATTERN.findall(text)))
        rules = list(set(self.RULE_PATTERN.findall(text)))
        return PolicyPassage(
            passage_id=passage_id,
            document=doc_name,
            page_number=page_num,
            section=section,
            text=text,
            rule_hints=rules,
            course_hints=courses,
        )

    def _fallback_paragraph_chunking(self, pages: List[Dict[str, Any]]) -> List[PolicyPassage]:
        passages = []
        p_count = 1
        for page in pages:
            paragraphs = [p.strip() for p in page["text"].split("\n\n") if p.strip()]
            for para in paragraphs:
                pid = f"P{p_count:03d}"
                courses = list(set(self.COURSE_PATTERN.findall(para)))
                rules = list(set(self.RULE_PATTERN.findall(para)))
                passages.append(PolicyPassage(
                    passage_id=pid,
                    document=page["document"],
                    page_number=page["page_number"],
                    section="General",
                    text=para,
                    rule_hints=rules,
                    course_hints=courses,
                ))
                p_count += 1
        return passages
