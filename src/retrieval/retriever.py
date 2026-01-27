# src/retrieval/retriever.py

from abc import ABC, abstractmethod
from typing import List


class BaseRetriever(ABC):
    """
    Retrieval strategy interface.
    """

    @abstractmethod
    def retrieve(self, query: str) -> List[str]:
        """
        Return list of context strings.
        """
        pass
