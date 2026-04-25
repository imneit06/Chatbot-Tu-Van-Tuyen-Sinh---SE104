from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain.retrievers.multi_vector import MultiVectorRetriever
except Exception:
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever


ID_KEY = "doc_id"

PARENTS_PATH = Path("data/processed/multivector_preview/parents.jsonl")
CHROMA_DIR = Path("storage/chroma")
COLLECTION_NAME = "uit_admission_multivector"

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda"  # Nếu lỗi GPU thì đổi thành "cpu"


def load_parent_docstore():
    parents = []

    with open(PARENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            doc = Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )

            parents.append((item["doc_id"], doc))

    docstore = InMemoryStore()
    docstore.mset(parents)

    return docstore


def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": DEVICE,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def build_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={
            "hnsw:space": "cosine",
        },
    )


def route_query(question: str):
    q = question.lower()

    curriculum_keywords = [
        "chương trình đào tạo",
        "khung chương trình",
        "môn học",
        "tín chỉ",
        "cơ sở ngành",
        "chuyên ngành",
        "chuẩn đầu ra",
        "mục tiêu đào tạo",
        "tốt nghiệp",
        "thực tập",
        "đồ án",
    ]

    admission_keywords = [
        "điểm chuẩn",
        "chỉ tiêu",
        "xét tuyển",
        "tổ hợp",
        "phương thức",
        "học bạ",
        "thpt",
        "đề án tuyển sinh",
        "đăng ký",
    ]

    if any(k in q for k in curriculum_keywords):
        return {"file_type": "html"}

    if any(k in q for k in admission_keywords):
        return {"file_type": "pdf"}

    return None


def build_retriever(metadata_filter=None, k=5):
    vectorstore = build_vectorstore()
    docstore = load_parent_docstore()

    search_kwargs = {
        "k": k,
    }

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
        search_kwargs=search_kwargs,
    )


def retrieve_docs(question: str, k=5):
    metadata_filter = route_query(question)

    retriever = build_retriever(
        metadata_filter=metadata_filter,
        k=k,
    )

    docs = retriever.invoke(question)

    return docs, metadata_filter


def format_context(docs):
    blocks = []

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata

        source = meta.get("source", "unknown")
        location = meta.get("location", "unknown")
        parent_type = meta.get("parent_type", "unknown")

        block = f"""
[DOCUMENT {i}]
Nguồn: {source}
Vị trí: {location}
Loại: {parent_type}

Nội dung:
{doc.page_content}
"""
        blocks.append(block)

    return "\n\n".join(blocks)


def format_sources(docs):
    sources = []

    for doc in docs:
        meta = doc.metadata

        sources.append(
            {
                "source": meta.get("source"),
                "location": meta.get("location"),
                "parent_type": meta.get("parent_type"),
            }
        )

    return sources