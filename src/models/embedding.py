import os
from typing import List, Literal
import numpy as np
from google import genai
from google.genai import types


EmbeddingTask = Literal[
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "SEMANTIC_SIMILARITY"
]


class GeminiEmbeddingModel:
    """
    Gemini Embedding Wrapper
    - Supports batch embedding
    - Explicit task_type
    - Returns np.ndarray
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "embedding-001",
        output_dimensionality: int | None = None,
        batch_size: int = 100
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is required")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Load from settings if available
        try:
            from ..config import settings
            self.output_dimensionality = output_dimensionality or settings.EMBEDDING_DIM
        except (ImportError, AttributeError):
            # Default to 1024 if settings not available
            self.output_dimensionality = output_dimensionality or 768

    def embed(
        self,
        texts: List[str],
        task_type: EmbeddingTask = "RETRIEVAL_DOCUMENT"
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.output_dimensionality), dtype=np.float32)

        embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.output_dimensionality
                )
            )

            for emb in response.embeddings:
                embeddings.append(emb.values)

        arr = np.asarray(embeddings, dtype=np.float32)

        if arr.shape[1] != self.output_dimensionality:
            raise ValueError(
                f"Embedding dim mismatch: "
                f"expected {self.output_dimensionality}, "
                f"got {arr.shape[1]}"
            )

        return arr


    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Convenience wrapper for document embedding"""
        return self.embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed single query → shape (1, dim)"""
        return self.embed([query], task_type="RETRIEVAL_QUERY")[0]
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension size"""
        return self.output_dimensionality
