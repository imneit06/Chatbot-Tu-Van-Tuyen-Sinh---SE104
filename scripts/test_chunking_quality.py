# scripts/test_chunking_quality.py
from pathlib import Path
from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]
from langchain_text_splitters import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.indexing.pipeline import extract_html_text_tables_images

test_html = list(Path("data/raw/html").glob("*.html"))[0]
text, tables, images = extract_html_text_tables_images(test_html)
text = text[:5000]

splitter_new = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=120,
    separators=["\n\n", "\n", "• ", ". ", "; ", ": ", ",", " "],
)

splitter_old = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
    separators=["\n\n", "\n", ".", " ", ""],
)

docs_new = splitter_new.split_documents([Document(page_content=text)])
docs_old = splitter_old.split_documents([Document(page_content=text)])

print(f"Config cũ: {len(docs_old)} chunks")
print(f"Config mới: {len(docs_new)} chunks")

for i, doc in enumerate(docs_new):
    if doc.page_content and doc.page_content[-1] not in ".\n":
        print(f"  Chunk {i} bị cắt giữa: '{doc.page_content[-30:]}'")

print("\n--- Config mới, chunk đầu tiên ---")
print(docs_new[0].page_content[:300])