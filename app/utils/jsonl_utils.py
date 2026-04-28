import json
from pathlib import Path
from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]

def doc_to_json(doc: Document):
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }

def write_jsonl(path: Path, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")