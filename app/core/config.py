from pathlib import Path
import os
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)


def get_env(key, default=None):
    return os.getenv(key, default)


def get_bool(key, default=False):
    value = os.getenv(key)

    if value is None:
        return default

    return value.strip().lower() in ["true", "1", "yes", "y"]


def get_int(key, default):
    value = os.getenv(key)

    if value is None:
        return default

    return int(value)


def get_float(key, default):
    value = os.getenv(key)

    if value is None:
        return default

    return float(value)


def get_path(key, default):
    value = os.getenv(key, default)
    path = Path(value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


# LLM
GOOGLE_API_KEY = get_env("GOOGLE_API_KEY")
GEMINI_MODEL = get_env("GEMINI_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = get_float("LLM_TEMPERATURE", 0.2)


# Embedding
EMBEDDING_MODEL_NAME = get_env("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DEVICE = get_env("EMBEDDING_DEVICE", "cuda")
NORMALIZE_EMBEDDINGS = get_bool("NORMALIZE_EMBEDDINGS", True)


# Paths
PDF_DIR = get_path("PDF_DIR", "data/raw/pdf")
HTML_DIR = get_path("HTML_DIR", "data/raw/html")
PROCESSED_DIR = get_path("PROCESSED_DIR", "data/processed/multivector_preview")
DOWNLOADED_IMAGES_DIR = get_path("DOWNLOADED_IMAGES_DIR", "data/processed/downloaded_images")
CHROMA_DIR = get_path("CHROMA_DIR", "storage/chroma")

PARENTS_PATH = PROCESSED_DIR / "parents.jsonl"
CHILDREN_PATH = PROCESSED_DIR / "children.jsonl"


# Chroma
CHROMA_COLLECTION_NAME = get_env(
    "CHROMA_COLLECTION_NAME",
    "uit_admission_multivector",
)

CHROMA_DISTANCE_METRIC = get_env("CHROMA_DISTANCE_METRIC", "cosine")
RESET_INDEX = get_bool("RESET_INDEX", True)
CHROMA_BATCH_SIZE = get_int("CHROMA_BATCH_SIZE", 512)


# Ingestion
PROCESS_PDF = get_bool("PROCESS_PDF", True)
PROCESS_HTML = get_bool("PROCESS_HTML", True)


# Chunking
CHUNK_SIZE = get_int("CHUNK_SIZE", 700)
CHUNK_OVERLAP = get_int("CHUNK_OVERLAP", 120)


# Retrieval
RETRIEVAL_TOP_K = get_int("RETRIEVAL_TOP_K", 5)
DEBUG_TOP_K = get_int("DEBUG_TOP_K", 8)

# Memory
MEMORY_DIR = get_path("MEMORY_DIR", "storage/memory")
MEMORY_MAX_RECENT_TURNS = get_int("MEMORY_MAX_RECENT_TURNS", 4)
MEMORY_SUMMARY_TRIGGER_TURNS = get_int("MEMORY_SUMMARY_TRIGGER_TURNS", 8)
QUESTION_REWRITE_ENABLED = get_bool("QUESTION_REWRITE_ENABLED", True)