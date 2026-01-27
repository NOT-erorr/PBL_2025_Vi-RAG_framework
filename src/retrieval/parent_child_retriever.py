from typing import List

from .qdrant import (
    retrieve_child_hits,
    extract_parent_ids_from_hits
)
from .docstore import fetch_parent_contexts


class ParentChildRetriever:
    """
    Standard Parent–Child Retrieval.

    child chunk → vector search
    parent chunk → context assembly
    """

    def __init__(
        self,
        *,
        vector_store,
        embedder,
        docstore,
        top_k: int = 5,
        max_parent_docs: int = 5
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.docstore = docstore

        self.top_k = top_k
        self.max_parent_docs = max_parent_docs

    def retrieve(self, query: str) -> List[str]:
        hits = retrieve_child_hits(
            vector_store=self.vector_store,
            embedder=self.embedder,
            query=query,
            top_k=self.top_k
        )

        parent_ids = extract_parent_ids_from_hits(hits)

        contexts = fetch_parent_contexts(
            docstore=self.docstore,
            parent_ids=parent_ids,
            max_docs=self.max_parent_docs
        )

        return contexts
