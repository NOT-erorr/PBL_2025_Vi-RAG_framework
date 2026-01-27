from dotenv import load_dotenv
import os

load_dotenv()  # tự tìm .env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
if GEMINI_API_KEY is None:
    raise RuntimeError("Missing GEMINI_API_KEY")
if QDRANT_API_KEY is None:
    raise RuntimeError("Missing QDRANT_API_KEY")
if QDRANT_URL is None:
    raise RuntimeError("Missing QDRANT_URL")

__all__ = ["GEMINI_API_KEY", "QDRANT_API_KEY", "QDRANT_URL"]
