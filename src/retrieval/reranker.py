class LLMReranker:

    def __init__(self, llm, top_k: int = 5):
        self.llm = llm
        self.top_k = top_k

    def rerank(self, query: str, contexts: list[str]) -> list[str]:
        scored = []

        for ctx in contexts:
            prompt = f"""
Query: {query}

Context:
{ctx}

Score relevance from 1 to 10:
"""
            score_text = self.llm.generate(
                query="",
                contexts=[prompt]
            )

            try:
                score = float(score_text.strip().split()[-1])
            except:
                score = 5

            scored.append((score, ctx))

        scored.sort(reverse=True, key=lambda x: x[0])

        return [ctx for _, ctx in scored[: self.top_k]]

class HybridRerankRetriever:

    def __init__(
        self,
        hybrid_retriever,
        reranker
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def retrieve(self, query: str):
        candidates = self.hybrid_retriever.retrieve(query)
        return self.reranker.rerank(query, candidates)
