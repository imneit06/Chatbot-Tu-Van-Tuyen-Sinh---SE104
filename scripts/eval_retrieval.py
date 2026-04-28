from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from app.rag.retriever import build_retriever, build_vectorstore
from app.core.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
    RETRIEVAL_TOP_K,
)


def load_eval_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_retriever_for_eval(k=None):
    vectorstore = build_vectorstore()
    docstore = _load_docstore()

    if k is None:
        k = RETRIEVAL_TOP_K

    search_kwargs = {
        "k": k,
    }

    try:
        from langchain.retrievers.multi_vector import MultiVectorRetriever  # pyright: ignore[reportMissingImports]
    except Exception:
        from langchain_classic.retrievers.multi_vector import MultiVectorRetriever  # pyright: ignore[reportMissingImports]

    ID_KEY = "doc_id"

    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
        search_kwargs=search_kwargs,
    )


def _load_docstore():
    from app.core.config import PARENTS_PATH
    from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]
    from langchain_core.stores import InMemoryStore  # pyright: ignore[reportMissingImports]

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


def get_file_type(doc):
    source = doc.metadata.get("source", "")
    if "pdf" in source.lower():
        return "pdf"
    if "html" in source.lower():
        return "html"
    return "unknown"


def evaluate_retrieval(dataset, k_values=None):
    if k_values is None:
        k_values = [3, 5, 10]

    if isinstance(dataset, str):
        dataset = load_eval_dataset(dataset)

    results = {
        "total": len(dataset),
        "per_question": [],
        "summary": {},
    }

    for k in k_values:
        results["summary"][k] = {
            "hit_rate": 0.0,
            "mrr": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "hit_count": 0,
        }

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        expected_file_types = set(item["expected_sources"])

        metadata_filter = route_query(question)

        retriever = build_retriever_for_eval(k=max(k_values))

        docs = retriever.invoke(question)

        retrieved_file_types = set(get_file_type(d) for d in docs)
        retrieved_sources = [d.metadata.get("source", "") for d in docs]

        hits = []
        for i, doc in enumerate(docs):
            file_type = get_file_type(doc)
            hit = file_type in expected_file_types
            hits.append(hit)

        rr = 0.0
        for i, hit in enumerate(hits):
            if hit:
                rr = 1.0 / (i + 1)
                break

        ndcg = 0.0
        if hits:
            dcg = sum((1.0 / np.log2(i + 2)) for i, h in enumerate(hits) if h)
            ideal_hits = [True] * min(len(expected_file_types), len(hits))
            idcg = sum((1.0 / np.log2(i + 2)) for i in range(len(ideal_hits)))
            ndcg = dcg / idcg if idcg > 0 else 0.0

        question_result = {
            "id": qid,
            "question": question,
            "retrieved": retrieved_sources,
            "hits": hits,
            "hit": any(hits),
            "rr": rr,
            "ndcg": ndcg,
            "retrieved_file_types": list(retrieved_file_types),
            "expected_file_types": list(expected_file_types),
        }

        for k in k_values:
            k_hits = hits[:k]
            k_hit = any(k_hits)
            k_relevant = sum(k_hits)
            k_total_relevant = len(expected_file_types)

            recall = k_relevant / k_total_relevant if k_total_relevant > 0 else 0.0
            precision = k_relevant / k if k > 0 else 0.0

            if k_hit:
                results["summary"][k]["hit_count"] += 1

            results["summary"][k]["hit_rate"] += (1.0 if k_hit else 0.0)
            results["summary"][k]["mrr"] += rr
            results["summary"][k]["recall"] += recall
            results["summary"][k]["precision"] += precision

        results["per_question"].append(question_result)

    n = len(dataset)
    for k in k_values:
        s = results["summary"][k]
        s["hit_rate"] /= n
        s["mrr"] /= n
        s["recall"] /= n
        s["precision"] /= n

    return results


def print_results(results, title="Retrieval Evaluation Results"):
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")
    print(f"Total questions: {results['total']}")

    print(f"\n{'K':<5} {'Hit Rate':<12} {'MRR':<12} {'Recall':<12} {'Precision':<12} {'Hits':<6}")
    print("-" * 65)

    for k, s in sorted(results["summary"].items()):
        print(
            f"{k:<5} "
            f"{s['hit_rate']:.3f}      "
            f"{s['mrr']:.3f}      "
            f"{s['recall']:.3f}      "
            f"{s['precision']:.3f}      "
            f"{s['hit_count']}/{results['total']}"
        )

    print(f"\n{'=' * 80}")
    print(" Per-question details:")
    print(f"{'=' * 80}")

    for q in results["per_question"]:
        status = "✓" if q["hit"] else "✗"
        print(f"\n{status} {q['id']} — {q['question']}")
        print(f"   Expected: {q['expected_file_types']} | Retrieved: {q['retrieved_file_types']}")
        print(f"   Hits: {q['hits']} | RR: {q['rr']:.3f} | NDCG: {q['ndcg']:.3f}")

        for i, src in enumerate(q["retrieved"]):
            marker = "→" if q["hits"][i] else "✗"
            print(f"      {marker} [{i+1}] {src}")


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_path}")
