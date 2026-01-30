"""Vi-RAG: Vietnamese Retrieval-Augmented Generation Framework

A comprehensive RAG framework specifically designed for Vietnamese language,
supporting PDF, TXT, and DOCX documents with hierarchical chunking and semantic search.
"""

__version__ = "0.1.2"
__author__ = "QuocLong"
__license__ = ""

# Note: Main components are available via submodules
# Example imports:
#   from vi_rag.core.document import Document, DocumentNode
#   from vi_rag.ingestion.loader import DocumentLoader
#   from vi_rag.models.embedding import GeminiEmbeddingModel
#   from vi_rag.retrieval.qdrant import QdrantVectorStore

__all__ = [
    "__version__",
    "__author__",
    "__license__",
]
