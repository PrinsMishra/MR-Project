"""
Converts Northbridge Institute of Technology (NITech) raw policy text into an official PDF document.
Uses PyMuPDF (fitz) to create a multi-page PDF with headers, footers, and passage boundaries.
"""

from pathlib import Path
import fitz  # PyMuPDF

SOURCE_TXT = Path(__file__).parent / "academic_policy_nitech.txt"
OUTPUT_PDF = Path(__file__).parent / "academic_policy_nitech.pdf"


def build_policy_pdf(txt_path: Path = SOURCE_TXT, pdf_path: Path = OUTPUT_PDF):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc = fitz.open()

    page_width = 595  # A4 width in points
    page_height = 842  # A4 height in points
    margin_x = 54
    margin_top = 60
    margin_bottom = 60
    line_height = 14

    current_y = margin_top
    page = doc.new_page(width=page_width, height=page_height)
    page_num = 1

    def draw_header_footer(p, p_num):
        # Header
        p.insert_text(
            fitz.Point(margin_x, 35),
            "NITech Academic Regulations Manual (2026-2027) | Document Ref: NIT-ACAD-REG-2026-V1",
            fontsize=8,
            color=(0.4, 0.4, 0.4),
        )
        # Footer
        p.insert_text(
            fitz.Point(page_width - margin_x - 60, page_height - 30),
            f"Page {p_num}",
            fontsize=8,
            color=(0.4, 0.4, 0.4),
        )

    draw_header_footer(page, page_num)

    for line in lines:
        text = line.rstrip()

        # Check page break
        if current_y > (page_height - margin_bottom):
            page_num += 1
            page = doc.new_page(width=page_width, height=page_height)
            draw_header_footer(page, page_num)
            current_y = margin_top

        if text.startswith("==="):
            current_y += 6
            continue
        elif text.startswith("SECTION") or text.startswith("NORTHBRIDGE"):
            current_y += 8
            page.insert_text(
                fitz.Point(margin_x, current_y),
                text,
                fontsize=11,
                fontname="helv",
                color=(0.1, 0.2, 0.5),
            )
            current_y += line_height + 4
        elif text.startswith("---"):
            current_y += 6
        elif text.startswith("[Passage"):
            # Passage rule title in bold-ish
            current_y += 4
            page.insert_text(
                fitz.Point(margin_x, current_y),
                text,
                fontsize=9.5,
                fontname="helv",
                color=(0.0, 0.0, 0.0),
            )
            current_y += line_height + 2
        else:
            if text.strip() == "":
                current_y += 6
            else:
                page.insert_text(
                    fitz.Point(margin_x + 10, current_y),
                    text,
                    fontsize=9,
                    fontname="helv",
                    color=(0.2, 0.2, 0.2),
                )
                current_y += line_height

    doc.save(pdf_path)
    doc.close()
    print(f"Generated PDF with {page_num} pages at: {pdf_path}")


if __name__ == "__main__":
    build_policy_pdf()
