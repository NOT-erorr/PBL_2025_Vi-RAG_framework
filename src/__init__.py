"""RAG System - Main Package"""

__version__ = "0.1.0"

# Import các module con để dễ dàng truy cập
from src import core
# from src import models
from src import ingestion
# from src import retrieval
# from src import plugin

__all__ = [
    "core",
    "models",
    "ingestion",
    "retrieval",
    "plugin",
]
