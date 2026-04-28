import pdfplumber  # pyright: ignore[reportMissingImports]
from pathlib import Path


def extract_pdf_text_tables_images(pdf_path: Path):
    """Trích xuất text + tables từ 1 PDF page."""
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            yield page_idx, text, tables