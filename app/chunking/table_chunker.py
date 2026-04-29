from langchain_core.documents import Document  

from ..utils.jsonl_utils import doc_to_json


ID_KEY = "doc_id"


def clean_cell(cell) -> str:
    if cell is None:
        return ""
    return str(cell).replace("\n", " ").strip()


def make_child_content_prefix(source, location, child_type, file_type, extra_note=""):
    prefix = [
        f"[Nguồn file] {source}",
        f"[File type] {file_type}",
        f"[Vị trí/trang] {location}",
        f"[Loại child] {child_type}",
    ]
    if extra_note:
        prefix.append(f"[Ghi chú] {extra_note}")
    return "\n".join(prefix) + "\n\n"


def table_to_markdown(table) -> str:
    lines = []
    for row in table:
        cleaned_row = [clean_cell(cell) for cell in row]
        lines.append("| " + " | ".join(cleaned_row) + " |")
    return "\n".join(lines)


def make_table_summary(table, file_name: str, location, table_idx: int) -> str:
    if not table:
        return ""
    header = [clean_cell(cell) for cell in table[0]]
    header_text = ", ".join([h for h in header if h])
    if not header_text:
        header_text = "không xác định rõ header"
    data_rows = len([r for r in table[1:] if any(clean_cell(c) for c in r)])
    sample_rows = []
    for row in table[1:4]:
        cells = [clean_cell(c) for c in row]
        if any(cells):
            sample_rows.append(" | ".join(cells))
    sample_text = "\n  ".join(sample_rows) if sample_rows else "không đọc được dòng nào"
    return (
        f"Bảng số {table_idx} trong file {file_name}, vị trí/trang {location}. "
        f"Các cột gồm: {header_text}. "
        f"Số dòng dữ liệu: {data_rows}. "
        f"Mẫu dữ liệu:\n  {sample_text}. "
        f"Bảng này có thể chứa thông tin tuyển sinh hoặc đào tạo như mã ngành, "
        f"tên ngành, chỉ tiêu, điểm chuẩn, học phí, phương thức xét tuyển, "
        f"tổ hợp môn, môn học, số tín chỉ hoặc ghi chú."
    )


def make_table_row_docs(
    table,
    parent_id: str,
    source: str,
    location,
    table_idx: int,
    file_type: str,
    group_size: int = 5,
    extra_metadata=None
):
    child_docs = []
    if not table or len(table) < 2:
        return child_docs

    header = [clean_cell(cell) for cell in table[0]]
    rows = table[1:]

    for group_start in range(0, len(rows), group_size):
        group_rows = rows[group_start:group_start + group_size]
        group_content_parts = []
        for row in group_rows:
            cells = [clean_cell(cell) for cell in row]
            pairs = []
            for h, c in zip(header, cells):
                if h or c:
                    label = h if h else "Cột không tên"
                    pairs.append(f"{label}: {c}")
            group_content_parts.append(" | ".join(pairs))

        group_content = " || ".join(group_content_parts)
        if not group_content.strip():
            continue

        row_start = group_start + 1
        row_end = min(group_start + group_size, len(rows))

        page_content = (
            make_child_content_prefix(source, location, "table_row_group", file_type)
            + f"Các dòng bảng tuyển sinh/đào tạo (dòng {row_start}-{row_end}): {group_content}"
        )

        metadata = {
            ID_KEY: parent_id,
            "source": source,
            "location": location,
            "child_type": "table_row_group",
            "table_index": table_idx,
            "row_start": row_start,
            "row_end": row_end,
            "file_type": file_type,
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        child_docs.append(
            Document(
                page_content=page_content,
                metadata=metadata
            )
        )

    return child_docs


def add_table_children(
    child_records,
    tables,
    parent_id: str,
    source: str,
    file_name: str,
    location,
    file_type: str,
    extra_metadata=None,
):
    table_blocks = []

    for table_idx, table in enumerate(tables, start=1):
        if not table:
            continue

        markdown_table = table_to_markdown(table)
        if markdown_table.strip():
            table_blocks.append(
                f"[BẢNG {table_idx} - {file_name} - vị trí/trang {location}]\n{markdown_table}"
            )

        summary = make_table_summary(table=table, file_name=file_name, location=location, table_idx=table_idx)

        if summary.strip():
            page_content = (
                make_child_content_prefix(
                    source=source,
                    location=location,
                    child_type="table_summary",
                    file_type=file_type,
                    extra_note="Tóm tắt bảng để hỗ trợ truy vấn RAG.",
                )
                + summary
            )
            summary_metadata = {
                ID_KEY: parent_id,
                "source": source,
                "location": location,
                "child_type": "table_summary",
                "table_index": table_idx,
                "file_type": file_type,
            }

            if extra_metadata:
                summary_metadata.update(extra_metadata)

            summary_doc = Document(
                page_content=page_content,
                metadata=summary_metadata
            )
            child_records.append(doc_to_json(summary_doc))

        row_docs = make_table_row_docs(
            table=table,
            parent_id=parent_id,
            source=source,
            location=location,
            table_idx=table_idx,
            file_type=file_type,
            extra_metadata=extra_metadata
        )
        for row_doc in row_docs:
            child_records.append(doc_to_json(row_doc))

    return table_blocks