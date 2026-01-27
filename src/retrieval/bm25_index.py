from rank_bm25 import BM25Okapi
import re


class BM25Index:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.tokenized = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized)

    def _tokenize(self, text: str):
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, top_k: int = 5):
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [idx for idx, _ in ranked[:top_k]]
