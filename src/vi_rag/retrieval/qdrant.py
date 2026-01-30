from typing import List, Dict, Any, Optional
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter
)

try:
    from ..config import settings
except ImportError:
    settings = None


class QdrantVectorStore:
    def __init__(
        self,
        collection_name: Optional[str] = None,
        vector_dim: Optional[int] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        distance: Distance = Distance.COSINE,
        api_key: Optional[str] = None,
        url: Optional[str] = None
    ):
        if settings:
            self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
            self.host = host or settings.QDRANT_HOST
            self.port = port or settings.QDRANT_PORT
            self.vector_dim = vector_dim or settings.QDRANT_VECTOR_DIM
            self.api_key = api_key or settings.QDRANT_API_KEY
            self.url = url or settings.QDRANT_URL
        else:
            self.collection_name = collection_name or "rag_documents"
            self.host = host or "localhost"
            self.port = port or 6333
            self.vector_dim = vector_dim or 768
            self.api_key = api_key
            self.url = url

        self.distance = distance
        self.client: Optional[QdrantClient] = None

    # -------------------------
    # Lifecycle methods
    # -------------------------
    def connect(self):
        if self.url:
            # Cloud / remote Qdrant
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=15.0
            )
        else:
            # Local / server mode
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                timeout=5.0
            )   

        try:
            self.client.get_collections()
        except Exception as e:
            raise RuntimeError(
                f"Qdrant server not reachable"
            ) from e

    def ensure_collection(self):
        assert self.client is not None, "Call connect() first"

        collections = self.client.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_dim,
                distance=self.distance
            )
        )

    # -------------------------
    # Data operations
    # -------------------------
    def add_vectors(
        self,
        vectors: np.ndarray,
        payloads: list[dict],
        ids: list[str] | None = None,
        batch_size: int = 64
    ):
        assert self.client is not None
        assert len(vectors) == len(payloads)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

        for i in range(0, len(vectors), batch_size):
            batch_points = [
                PointStruct(
                    id=pid,
                    vector=vec.tolist(),
                    payload=payload
                )
                for pid, vec, payload in zip(
                    ids[i:i+batch_size],
                    vectors[i:i+batch_size],
                    payloads[i:i+batch_size]
                )
            ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=batch_points,
            wait=True
        )

    def search(self, query_vector, limit=5, filter=None):
        assert self.client is not None, "Call connect() first"

        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
            query_filter=filter
        ).points
