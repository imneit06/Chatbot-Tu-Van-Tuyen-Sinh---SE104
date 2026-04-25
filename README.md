# UIT Admission Chatbot

Đồ án cuối kỳ môn **SE104 - Nhập môn Công nghệ Phần mềm** tại **UIT**.

Nguyễn Xuân Tiến - 24521778
Phan Tấn Tiến - 24521780
Phạm Hồ Hữu Trí - 24521841

## Giới thiệu

Hệ thống chatbot hỗ trợ tư vấn tuyển sinh, cho phép:

- tra cứu ngành học
- tra cứu học phí
- tra cứu phương thức xét tuyển
- tra cứu tổ hợp môn
- tra cứu điểm chuẩn
- hỏi đáp từ tài liệu tuyển sinh bằng RAG

---

## Cấu trúc project

```bash
uit-admission-rag/
│
├── app/                               # code chính backend
│   ├── api/                           # FastAPI / Flask endpoints
│   │   ├── chat.py                    # /chat
│   │   ├── ingest.py                  # /admin/ingest
│   │   ├── reindex.py                 # /admin/reindex
│   │   ├── documents.py               # list docs, delete doc, status
│   │   └── health.py
│   │
│   ├── core/
│   │   ├── config.py                  # đọc env, settings
│   │   ├── logging.py                 # logger
│   │   ├── constants.py               # tên folder, ext hỗ trợ
│   │   └── prompts.py                 # system prompt, citation prompt
│   │
│   ├── loaders/                       # đọc dữ liệu đầu vào
│   │   ├── base_loader.py
│   │   ├── pdf_loader.py
│   │   ├── txt_loader.py
│   │   ├── html_loader.py
│   │   └── loader_router.py           # tự chọn loader theo extension
│   │
│   ├── preprocess/
│   │   ├── clean_text.py              # bỏ ký tự rác, normalize space
│   │   ├── metadata_extractor.py      # title, year, page, source_type...
│   │   ├── deduplicate.py             # loại bản trùng
│   │   └── normalize_document.py      # chuẩn hóa về 1 schema chung
│   │
│   ├── chunking/
│   │   ├── chunker.py                 # chia chunk chính
│   │   ├── heading_splitter.py        # chia theo tiêu đề trước
│   │   └── chunk_validator.py         # kiểm tra chunk quá ngắn/dài
│   │
│   ├── embeddings/
│   │   ├── embedding_factory.py       # chọn model embedding
│   │   └── embed_documents.py
│   │
│   ├── vectorstores/
│   │   ├── faiss_store.py
│   │   ├── chroma_store.py            # để sẵn nếu sau này đổi
│   │   └── store_factory.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py               # similarity search
│   │   ├── filters.py                 # lọc theo year/type/program
│   │   ├── reranker.py                # optional
│   │   └── hybrid_retriever.py        # optional về sau
│   │
│   ├── chains/
│   │   ├── rag_chain.py               # chain hỏi đáp chính
│   │   ├── answer_formatter.py        # format answer + nguồn
│   │   └── citation_builder.py        # gom page/source cho output
│   │
│   ├── services/
│   │   ├── ingest_service.py
│   │   ├── reindex_service.py
│   │   ├── query_service.py
│   │   └── document_service.py
│   │
│   ├── schemas/
│   │   ├── chat.py                    # request/response models
│   │   ├── document.py
│   │   └── admin.py
│   │
│   └── utils/
│       ├── file_hash.py               # hash file để detect update
│       ├── paths.py
│       ├── timers.py
│       └── helpers.py
│
├── data/
│   ├── raw/
│   │   ├── pdf/
│   │   ├── txt/
│   │   └── html/
│   │
│   ├── interim/
│   │   ├── parsed/                    # text sau khi load
│   │   ├── cleaned/                   # text sau khi làm sạch
│   │   └── manifests/                 # metadata tổng hợp từng file
│   │
│   ├── processed/
│   │   ├── chunks/                    # chunk JSONL / parquet
│   │   ├── embeddings/                # optional cache embedding
│   │   └── reports/                   # report ingest
│   │
│   └── archive/                       # file cũ / đã thay thế
│
├── storage/
│   ├── vectorstore/
│   │   ├── faiss/                     # index FAISS
│   │   └── chroma/                    # nếu dùng Chroma
│   │
│   ├── docstore/                      # mapping doc_id <-> metadata
│   └── cache/
│
├── scripts/
│   ├── ingest_once.py                 # ingest 1 lần
│   ├── reindex_all.py                 # rebuild index toàn bộ
│   ├── sync_data.py                   # copy data từ nguồn khác
│   ├── validate_chunks.py
│   └── evaluate_rag.py
│
├── tests/
│   ├── test_loaders.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_rag_chain.py
│   └── test_api.py
│
├── notebooks/
│   ├── inspect_pdf.ipynb
│   ├── chunk_experiments.ipynb
│   └── retrieval_debug.ipynb
│
├── frontend/                          # nếu bạn làm UI riêng
│   ├── src/
│   └── public/
│
├── .env
├── .env.example
├── requirements.txt
├── README.md
└── run.py