from dotenv import load_dotenv
import os
from pathlib import Path
from src.secret import *

# ======================================================
# Load environment
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

# ======================================================
# Helper
# ======================================================

def get_env(name: str, default=None, required: bool = False):
    value = os.getenv(name, default)

    if required and value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


# ======================================================
# Core settings
# ======================================================

ENV = get_env("ENV", "development")

GEMINI_API_KEY = get_env("GEMINI_API_KEY", required=True)

EMBEDDING_MODEL = get_env(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_TOP_K = int(get_env("VECTOR_TOP_K", 5))

# Embedding Configuration
EMBEDDING_DIM = int(get_env("EMBEDDING_DIM", 768))

# Qdrant Configuration
QDRANT_HOST = get_env("QDRANT_HOST", "localhost")
QDRANT_PORT = int(get_env("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = get_env("QDRANT_COLLECTION_NAME", "rag_documents")
QDRANT_VECTOR_DIM = int(get_env("QDRANT_VECTOR_DIM", 768))
QDRANT_API_KEY = get_env("QDRANT_API_KEY", required=True)
QDRANT_URL = get_env("QDRANT_URL", required=True)

# ======================================================
# Export
# ======================================================

__all__ = [
    "ENV",
    "GEMINI_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "VECTOR_TOP_K",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_COLLECTION_NAME",
    "QDRANT_VECTOR_DIM",
]
