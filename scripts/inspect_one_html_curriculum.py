from pathlib import Path
import json
import argparse
from collections import Counter


PARENTS_PATH = Path("data/processed/multivector_preview/parents.jsonl")
CHILDREN_PATH = Path("data/processed/multivector_preview/children.jsonl")
OUT_DIR = Path("data/processed/multivector_preview/inspect_one_html")


CHECK_KEYWORDS = [
    "mục tiêu đào tạo",
    "chuẩn đầu ra",
    "vị trí làm việc",
    "khối kiến thức",
    "khung chương trình",
    "chương trình đào tạo",
    "đại cương",
    "cơ sở ngành",
    "chuyên ngành",
    "tự chọn",
    "tốt nghiệp",
    "môn học",
    "tín chỉ",
    "thực tập",
    "đồ án",
]


def load_jsonl(path: Path):
    items = []

    if not path.exists():
        print(f"Không tìm thấy file: {path}")
        return items

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def norm(s: str) -> str:
    return str(s or "").lower()


def match_source(item, source_keyword: str):
    meta = item.get("metadata", {})
    source = norm(meta.get("source", ""))
    return norm(source_keyword) in source


def is_html_item(item):
    return item.get("metadata", {}).get("file_type") == "html"


def get_doc_id(item):
    meta = item.get("metadata", {})
    return item.get("doc_id") or meta.get("doc_id")


def format_parent(item, idx):
    meta = item.get("metadata", {})
    content = item.get("page_content", "")

    lines = []
    lines.append("=" * 120)
    lines.append(f"HTML PARENT #{idx}")
    lines.append("=" * 120)
    lines.append(f"doc_id      : {item.get('doc_id')}")
    lines.append(f"source      : {meta.get('source')}")
    lines.append(f"file_type   : {meta.get('file_type')}")
    lines.append(f"parent_type : {meta.get('parent_type')}")
    lines.append(f"location    : {meta.get('location')}")
    lines.append("-" * 120)
    lines.append("PARENT CONTENT:")
    lines.append(content)
    lines.append("")
    return "\n".join(lines)


def format_child(item, idx):
    meta = item.get("metadata", {})
    content = item.get("page_content", "")

    lines = []
    lines.append("=" * 120)
    lines.append(f"HTML CHILD #{idx}")
    lines.append("=" * 120)
    lines.append(f"child_type       : {meta.get('child_type')}")
    lines.append(f"doc_id           : {meta.get('doc_id')}")
    lines.append(f"source           : {meta.get('source')}")
    lines.append(f"location         : {meta.get('location')}")
    lines.append(f"table_index      : {meta.get('table_index')}")
    lines.append(f"row_index        : {meta.get('row_index')}")
    lines.append(f"image_index      : {meta.get('image_index')}")
    lines.append(f"image_src        : {meta.get('image_src')}")
    lines.append(f"image_local_path : {meta.get('image_local_path')}")
    lines.append("-" * 120)
    lines.append("CHILD CONTENT:")
    lines.append(content)
    lines.append("")
    return "\n".join(lines)


def build_keyword_report(text: str):
    lower_text = norm(text)

    lines = []
    lines.append("=" * 120)
    lines.append("CHECKLIST NỘI DUNG CHƯƠNG TRÌNH ĐÀO TẠO")
    lines.append("=" * 120)

    for kw in CHECK_KEYWORDS:
        status = "CÓ" if kw in lower_text else "KHÔNG THẤY"
        lines.append(f"{status:10} | {kw}")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-keyword",
        required=True,
        help="Một phần tên file HTML cần inspect, ví dụ: 'Công nghệ Thông tin' hoặc 'Khoa học Máy tính'",
    )

    parser.add_argument(
        "--content-keyword",
        default="",
        help="Nếu muốn lọc sâu trong child content, ví dụ: 'cơ sở ngành' hoặc 'khung chương trình'",
    )

    args = parser.parse_args()

    parents = load_jsonl(PARENTS_PATH)
    children = load_jsonl(CHILDREN_PATH)

    html_parents = [
        item for item in parents
        if is_html_item(item) and match_source(item, args.source_keyword)
    ]

    matched_parent_ids = {item.get("doc_id") for item in html_parents}

    html_children = [
        item for item in children
        if is_html_item(item)
        and (
            match_source(item, args.source_keyword)
            or get_doc_id(item) in matched_parent_ids
        )
    ]

    if args.content_keyword:
        ck = norm(args.content_keyword)
        html_children = [
            item for item in html_children
            if ck in norm(item.get("page_content", ""))
        ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = args.source_keyword.replace("/", "_").replace(" ", "_")
    parent_out = OUT_DIR / f"{safe_name}_parents.txt"
    child_out = OUT_DIR / f"{safe_name}_children.txt"
    report_out = OUT_DIR / f"{safe_name}_report.txt"

    all_text = "\n\n".join(
        [p.get("page_content", "") for p in html_parents]
        + [c.get("page_content", "") for c in html_children]
    )

    child_type_counter = Counter(
        item.get("metadata", {}).get("child_type", "unknown")
        for item in html_children
    )

    with open(parent_out, "w", encoding="utf-8") as f:
        for idx, item in enumerate(html_parents, start=1):
            f.write(format_parent(item, idx))
            f.write("\n")

    with open(child_out, "w", encoding="utf-8") as f:
        for idx, item in enumerate(html_children, start=1):
            f.write(format_child(item, idx))
            f.write("\n")

    with open(report_out, "w", encoding="utf-8") as f:
        f.write(build_keyword_report(all_text))
        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("THỐNG KÊ\n")
        f.write("=" * 120 + "\n")
        f.write(f"Số HTML parent match: {len(html_parents)}\n")
        f.write(f"Số HTML child match : {len(html_children)}\n\n")
        f.write("Child type:\n")
        for k, v in child_type_counter.items():
            f.write(f"- {k}: {v}\n")

    print("=" * 100)
    print("Inspect HTML curriculum xong")
    print("=" * 100)
    print(f"Source keyword      : {args.source_keyword}")
    print(f"Content keyword     : {args.content_keyword or '(không lọc)'}")
    print(f"Số HTML parent match: {len(html_parents)}")
    print(f"Số HTML child match : {len(html_children)}")

    print("\nChild type:")
    for k, v in child_type_counter.items():
        print(f"- {k}: {v}")

    print("\nFile report:")
    print(report_out)

    print("\nFile parent:")
    print(parent_out)

    print("\nFile children:")
    print(child_out)

    print("\nPreview checklist:")
    print(build_keyword_report(all_text))


if __name__ == "__main__":
    main()