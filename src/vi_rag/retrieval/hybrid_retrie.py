from .parent_child_retriever import ParentChildRetriever


class HybridRetriever:
    """
    Vector + BM25 hybrid retrieval
    """

    def __init__(
        self,
        *,
        vector_retriever: ParentChildRetriever,
        bm25_index,
        doc_texts: list[str],
        alpha: float = 0.5,
        top_k: int = 5
    ):
        """
        alpha: weight between semantic vs lexical
        """
        self.vector_retriever = vector_retriever
        self.bm25_index = bm25_index
        self.doc_texts = doc_texts
        self.alpha = alpha
        self.top_k = top_k

    def retrieve(self, query: str):
        # semantic
        semantic_contexts = self.vector_retriever.retrieve(query)

        # lexical
        bm25_ids = self.bm25_index.search(query, self.top_k)
        bm25_contexts = [self.doc_texts[i] for i in bm25_ids]

        # merge + deduplicate
        merged = []
        seen = set()

        for ctx in semantic_contexts + bm25_contexts:
            if ctx not in seen:
                seen.add(ctx)
                merged.append(ctx)

        return merged[: self.top_k]
