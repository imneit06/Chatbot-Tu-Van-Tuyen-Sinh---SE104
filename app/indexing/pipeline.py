from pathlib import Path
import uuid
import re
from langchain_core.documents import Document # pyright: ignore[reportMissingImports]
from langchain_text_splitters import RecursiveCharacterTextSplitter # pyright: ignore[reportMissingImports]

from app.core.config import (
    PROCESS_PDF,
    PROCESS_HTML,
    PDF_DIR,
    HTML_DIR,
    PARENTS_PATH,
    CHILDREN_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.core.config import PROCESSED_DIR, DOWNLOADED_IMAGES_DIR

from .utils.file_utils import mkdir # pyright: ignore[reportMissingImports]
from .utils.jsonl_utils import write_jsonl # pyright: ignore[reportMissingImports]
from .loaders.html_loader import extract_html_text_tables_images # pyright: ignore[reportMissingImports]
from .chunking.text_chunker import add_text_chunks # pyright: ignore[reportMissingImports]
from .chunking.table_chunker import add_table_children # pyright: ignore[reportMissingImports]
from .chunking.image_chunker import add_image_children # pyright: ignore[reportMissingImports]


ID_KEY = "doc_id"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Parent helpers
# =========================

def add_parent(
    parent_records,
    parent_id: str,
    content: str,
    source: str,
    location,
    parent_type: str,
    file_type: str,
):
    if not content.strip():
        return
    parent_records.append({
        "doc_id": parent_id,
        "page_content": content,
        "metadata": {
            "source": source,
            "location": location,
            "parent_type": parent_type,
            "file_type": file_type,
        },
    })


# =========================
# HTML helpers (section splitting)
# =========================

def is_section_heading(line: str) -> bool:
    line = line.strip()
    lower = line.lower()
    if not line:
        return False
    heading_keywords = [
        "giới thiệu chung", "mục tiêu đào tạo", "đối tượng tuyển sinh",
        "quy chế đào tạo", "chuẩn đầu ra", "chương trình đào tạo",
        "tỷ lệ các khối kiến thức", "khung chương trình", "khối kiến thức",
        "cơ sở ngành", "chuyên ngành", "tốt nghiệp", "thực tập", "đồ án",
    ]
    if any(keyword in lower for keyword in heading_keywords):
        return True
    if re.match(r"^\d+(\.\d+)*\.?\s+.{3,}$", line):
        return True
    letters = [c for c in line if c.isalpha()]
    if len(line) >= 8 and letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio >= 0.7:
            return True
    return False


def guess_major_name(text: str, source: str = "") -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:30]:
        lower = line.lower()
        if "cử nhân ngành" in lower or "kỹ sư ngành" in lower:
            return line
        if line.lower().startswith("ngành "):
            return line
    name = Path(source).stem
    return name


def split_text_into_sections(text: str, min_section_chars: int = 300):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections = []
    current_title = "Nội dung chính"
    current_lines = []

    for line in lines:
        if is_section_heading(line) and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({"title": current_title, "content": content})
            current_title = line
            current_lines = [line]
        else:
            if not current_lines and is_section_heading(line):
                current_title = line
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append({"title": current_title, "content": content})

    # gộp section quá ngắn vào section trước
    merged = []
    for sec in sections:
        if merged and len(sec["content"]) < min_section_chars:
            merged[-1]["content"] += "\n" + sec["content"]
        else:
            merged.append(sec)

    if not merged:
        return [{"title": "Nội dung chính", "content": text}]
    return merged


# =========================
# Process PDF
# =========================

def process_pdfs(parent_records, child_records, splitter):
    from .loaders.pdf_loader import extract_pdf_text_tables_images # pyright: ignore[reportMissingImports]

    pdf_files = list(PDF_DIR.rglob("*.pdf"))
    for pdf_path in pdf_files:
        print(f"Đang xử lý PDF: {pdf_path}")

        # inline clean_text
        def clean_text(t):
            lines = []
            for line in t.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
            return "\n".join(lines)

        for page_idx, text, tables in extract_pdf_text_tables_images(pdf_path):
            parent_id = str(uuid.uuid4())
            text = clean_text(text)

            table_blocks = add_table_children(
                child_records=child_records,
                tables=tables,
                parent_id=parent_id,
                source=str(pdf_path),
                file_name=pdf_path.name,
                location=page_idx,
                file_type="pdf",
            )

            parent_content = text + "\n\n" + "\n\n".join(table_blocks)

            add_parent(
                parent_records=parent_records,
                parent_id=parent_id,
                content=parent_content,
                source=str(pdf_path),
                location=page_idx,
                parent_type="pdf_page",
                file_type="pdf",
            )

            add_text_chunks(
                child_records=child_records,
                splitter=splitter,
                text=text,
                parent_id=parent_id,
                source=str(pdf_path),
                location=page_idx,
                file_type="pdf",
            )


# =========================
# Process HTML
# =========================

def process_htmls(parent_records, child_records, splitter):
    html_files = list(HTML_DIR.rglob("*.html")) + list(HTML_DIR.rglob("*.htm"))

    for html_path in html_files:
        print(f"Đang xử lý HTML: {html_path}")
        text, tables, images = extract_html_text_tables_images(html_path)

        major_name = guess_major_name(text, str(html_path))
        sections = split_text_into_sections(text)

        print(f"[HTML] Major guess: {major_name}")
        print(f"[HTML] Số section text: {len(sections)}")
        print(f"[HTML] Số bảng: {len(tables)}")
        print(f"[HTML] Số ảnh: {len(images)}")

        # 1. Mỗi section text = 1 parent
        for section_idx, section in enumerate(sections, start=1):
            parent_id = str(uuid.uuid4())
            section_title = section["title"]
            section_content = section["content"]
            parent_content = f"[SECTION {section_idx}] {section_title}\n\n{section_content}"

            add_parent(
                parent_records=parent_records,
                parent_id=parent_id,
                content=parent_content,
                source=str(html_path),
                location=f"html_section_{section_idx}",
                parent_type="html_section",
                file_type="html",
            )
            add_text_chunks(
                child_records=child_records,
                splitter=splitter,
                text=parent_content,
                parent_id=parent_id,
                source=str(html_path),
                location=f"html_section_{section_idx}",
                file_type="html",
                extra_metadata={
                    "doc_type": "curriculum",
                    "major_name": major_name,
                    "section_title": section_title,
                    "section_index": section_idx,
                },
            )

        # 2. Mỗi bảng HTML = 1 parent riêng
        for table_idx, table in enumerate(tables, start=1):
            parent_id = str(uuid.uuid4())
            table_blocks = add_table_children(
                child_records=child_records,
                tables=[table],
                parent_id=parent_id,
                source=str(html_path),
                file_name=html_path.name,
                location=f"html_table_{table_idx}",
                file_type="html",
            )
            if not table_blocks:
                continue
            parent_content = (
                f"[TABLE SECTION] Bảng {table_idx} trong {html_path.name}\n"
                f"[Ngành] {major_name}\n\n" + "\n\n".join(table_blocks)
            )
            add_parent(
                parent_records=parent_records,
                parent_id=parent_id,
                content=parent_content,
                source=str(html_path),
                location=f"html_table_{table_idx}",
                parent_type="html_table",
                file_type="html",
            )

        # 3. Mỗi ảnh OCR liên quan = 1 parent riêng
        for image_idx, image in enumerate(images, start=1):
            parent_id = str(uuid.uuid4())
            image_blocks = add_image_children(
                child_records=child_records,
                images=[image],
                parent_id=parent_id,
                source=str(html_path),
                file_name=html_path.name,
                location=f"html_image_{image_idx}",
                file_type="html",
            )
            if not image_blocks:
                continue
            parent_content = (
                f"[IMAGE SECTION] Hình {image_idx} trong {html_path.name}\n"
                f"[Ngành] {major_name}\n\n" + "\n\n".join(image_blocks)
            )
            add_parent(
                parent_records=parent_records,
                parent_id=parent_id,
                content=parent_content,
                source=str(html_path),
                location=f"html_image_{image_idx}",
                parent_type="html_image",
                file_type="html",
            )


# =========================
# Main
# =========================

def run_prepare_multivector():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n", "\n", ". ", "; ", ": ", ",",
            " ", "",
        ]
    )

    parent_records = []
    child_records = []

    if PROCESS_PDF:
        process_pdfs(parent_records, child_records, splitter)

    if PROCESS_HTML:
        process_htmls(parent_records, child_records, splitter)

    write_jsonl(PARENTS_PATH, parent_records)
    write_jsonl(CHILDREN_PATH, child_records)

    print("Xong.")
    print(f"Số parent docs: {len(parent_records)}")
    print(f"Số child docs: {len(child_records)}")
    print(f"Parent lưu tại: {PARENTS_PATH}")
    print(f"Child lưu tại: {CHILDREN_PATH}")


if __name__ == "__main__":
    run_prepare_multivector()