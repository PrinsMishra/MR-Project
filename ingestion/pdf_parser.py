"""
PyMuPDF (fitz) PDF Parser for Academic Policy Documents.
Extracts text with page numbers, bounding boxes, and structural layout.
"""

from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF


class PDFParser:
    """Extracts raw text and per-page layout metadata from PDF files."""

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

    def extract_pages(self) -> List[Dict[str, Any]]:
        """
        Extracts pages with page index and plain text content.
        Returns a list of dicts: [{'page_number': 1, 'text': '...'}]
        """
        doc = fitz.open(self.pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({
                "page_number": i + 1,
                "text": text,
                "document": self.pdf_path.name,
                "num_rects": len(page.get_drawings()),
            })
        doc.close()
        return pages
