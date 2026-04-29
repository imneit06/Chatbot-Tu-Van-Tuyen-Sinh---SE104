from pathlib import Path
import json
import shutil

from langchain_core.documents import Document  
from langchain_core.stores import InMemoryStore  
from langchain_chroma import Chroma  
from langchain_huggingface import HuggingFaceEmbeddings  

try:
    from langchain.retrievers.multi_vector import MultiVectorRetriever  
except Exception:
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever  

from app.core.config import (
    PARENTS_PATH,
    CHILDREN_PATH,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
    RESET_INDEX,
    CHROMA_BATCH_SIZE,
)

ID_KEY = "doc_id"


def load_parents():
    parents = []

    with open(PARENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            doc_id = item["doc_id"]

            doc = Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )

            parents.append((doc_id, doc))

    return parents


def load_children():
    children = []

    with open(CHILDREN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            doc = Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )

            if ID_KEY not in doc.metadata:
                continue

            if not doc.page_content.strip():
                continue

            children.append(doc)

    return children


def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": EMBEDDING_DEVICE,
        },
        encode_kwargs={
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        },
    )


def build_vectorstore():
    embeddings = build_embeddings()

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={
            "hnsw:space": CHROMA_DISTANCE_METRIC,
        },
    )

    return vectorstore


def build_retriever(vectorstore, docstore, metadata_filter=None, k=6):
    search_kwargs = {
        "k": k,
    }

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
        search_kwargs=search_kwargs,
    )

    return retriever


def add_documents_with_fallback(vectorstore, documents):
    """
    Add documents vào Chroma.
    Nếu số lượng vượt max batch của Chroma thì tự chia batch nhỏ.
    """
    try:
        vectorstore.add_documents(documents)
        return

    except Exception as e:
        message = str(e)

        if "batch" not in message.lower():
            raise e

        print("Chroma báo batch quá lớn, chuyển sang add theo batch nhỏ...")
        print("Lỗi gốc:", message)

    total = len(documents)

    for start in range(0, total, CHROMA_BATCH_SIZE):
        end = start + CHROMA_BATCH_SIZE
        batch = documents[start:end]

        print(f"Đang add batch {start} → {min(end, total)} / {total}")

        vectorstore.add_documents(batch)
        

def ingest():
    if RESET_INDEX and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    parents = load_parents()
    children = load_children()

    print(f"Số parent docs: {len(parents)}")
    print(f"Số child docs: {len(children)}")

    docstore = InMemoryStore()
    docstore.mset(parents)

    vectorstore = build_vectorstore()

    if children:
        print("Đang embedding children và lưu vào Chroma...")
        add_documents_with_fallback(vectorstore, children)

    print("Ingest xong.")

    return vectorstore, docstore


def debug_child_search(vectorstore, question: str, k=8, metadata_filter=None):
    print("\n" + "#" * 100)
    print("DEBUG CHILD SEARCH")
    print("Câu hỏi:", question)
    print("Filter:", metadata_filter)
    print("#" * 100)

    results = vectorstore.similarity_search_with_score(
        question,
        k=k,
        filter=metadata_filter,
    )

    for i, (doc, score) in enumerate(results, start=1):
        print("\n" + "-" * 100)
        print(f"Child {i}")
        print("Score:", score)
        print("Metadata:", doc.metadata)
        print("-" * 100)
        print(doc.page_content[:1500])


def test_retrieve(vectorstore, docstore):
    test_cases = [
        {
            "question": "Trong chương trình đào tạo ngành Khoa học Máy tính, các môn học cơ sở ngành gồm những gì?",
            "filter": {"file_type": "html"},
        },
        {
            "question": "Khung chương trình đào tạo gồm những khối kiến thức nào?",
            "filter": {"file_type": "html"},
        },
        {
            "question": "Tổng số tín chỉ tối thiểu là bao nhiêu?",
            "filter": {"file_type": "html"},
        },
        {
            "question": "Các phương thức xét tuyển gồm những gì?",
            "filter": {"file_type": "pdf"},
        },
        {
            "question": "Điểm chuẩn ngành Công nghệ thông tin là bao nhiêu?",
            "filter": {"file_type": "pdf"},
        },
    ]

    for case in test_cases:
        question = case["question"]
        metadata_filter = case["filter"]

        print("\n" + "=" * 100)
        print("Câu hỏi:", question)
        print("Filter:", metadata_filter)

        debug_child_search(
            vectorstore=vectorstore,
            question=question,
            k=8,
            metadata_filter=metadata_filter,
        )

        retriever = build_retriever(
            vectorstore=vectorstore,
            docstore=docstore,
            metadata_filter=metadata_filter,
            k=6,
        )

        docs = retriever.invoke(question)

        print("\nSố parent docs retrieve được:", len(docs))

        for i, doc in enumerate(docs, start=1):
            print("\n" + "-" * 100)
            print(f"Parent {i}")
            print("Metadata:", doc.metadata)
            print("-" * 100)
            print(doc.page_content[:2500])


def main():
    vectorstore, docstore = ingest()
    test_retrieve(vectorstore, docstore)


if __name__ == "__main__":
    main()