from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class RAGResponse:
    answer: str
    contexts: List[str]


class RAGEngine:

    def __init__(
        self,
        *,
        retriever,
        llm
    ):
        self.retriever = retriever
        self.llm = llm

    # -----------------------------
    # hooks (override if needed)
    # -----------------------------

    def before_query(self, query: str) -> str:
        return query

    def after_retrieve(self, contexts: List[str]) -> List[str]:
        return contexts

    def after_generate(self, answer: str) -> str:
        return answer

    # -----------------------------
    # core
    # -----------------------------

    def retrieve(self, query: str) -> List[str]:
        return self.retriever.retrieve(query)

    def generate(self, query: str) -> RAGResponse:
        query = self.before_query(query)

        contexts = self.retrieve(query)
        contexts = self.after_retrieve(contexts)

        answer = self.llm.generate(
            query=query,
            contexts=contexts
        )

        answer = self.after_generate(answer)

        return RAGResponse(
            answer=answer,
            contexts=contexts
        )

    def query(self, query: str) -> RAGResponse:
        return self.generate(query)
