"""
Plugin Module

This module contains plugin implementations for the RAG system.

Available Plugins:
- FaissVectorStore: FAISS-based vector store
- ParentDocumentStore: Parent document retrieval strategy

Example Usage:
    >>> from plugin.FaissVectorStore import FaissVectorStore
    >>> from plugin.ParentDocumentStore import ParentDocumentStore
    
    Or import everything:
    >>> from plugin import ParentDocumentStore, FaissVectorStore
"""

# Vector Store Plugins
from .FaissVectorStore import FaissVectorStore

# Parent Document Store Plugin
from .ParentDocumentStore import (
    ParentChildChunk,
    ParentDocumentStore,
    ParentChildTextSplitter
)

__all__ = [
    # Vector Stores
    'FaissVectorStore',
    
    # Parent Document Store
    'ParentChildChunk',
    'ParentDocumentStore',
    'ParentChildTextSplitter',
]

__version__ = '1.0.0'
