from langchain_core.documents import Document  

from ..utils.jsonl_utils import doc_to_json
from ..loaders.html_helpers import make_child_content_prefix


ID_KEY = "doc_id"


def add_text_chunks(
    child_records,
    splitter,
    text: str,
    parent_id: str,
    source: str,
    location,
    file_type: str,
    extra_metadata=None,
):
    if not text.strip():
        return

    metadata = {
        ID_KEY: parent_id,
        "source": source,
        "location": location,
        "child_type": "text_chunk",
        "file_type": file_type,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    text_docs = splitter.split_documents(
        [Document(page_content=text, metadata=metadata)]
    )

    for doc in text_docs:
        prefix = make_child_content_prefix(
            source=source,
            location=location,
            child_type="text_chunk",
            file_type=file_type,
            extra_note="Đoạn văn bản được tách từ một section của PDF/HTML.",
        )

        chunk_size = len(doc.page_content)
        if chunk_size < 300:
            chunk_quality = "short_fragment"
        elif chunk_size > 1000:
            chunk_quality = "long_chunk"
        else:
            chunk_quality = "good"

        doc.page_content = prefix + doc.page_content
        doc.metadata["chunk_quality"] = chunk_quality
        doc.metadata["chunk_char_count"] = chunk_size

        child_records.append(doc_to_json(doc))