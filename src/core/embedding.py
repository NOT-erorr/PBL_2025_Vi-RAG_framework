from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict
import numpy as np


class BaseEmbedding(ABC):
    """Abstract base class cho embedding models"""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed một list các documents
        
        Args:
            texts: Danh sách texts cần embed
            
        Returns:
            List[List[float]]: Danh sách embeddings
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed một query text
        
        Args:
            text: Query text cần embed
            
        Returns:
            List[float]: Query embedding
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Trả về dimension của embeddings"""
        pass


class BGEM3Embedding(BaseEmbedding):
    """
    BGE-M3 Embedding model - Multi-lingual, Multi-functionality, Multi-granularity
    Support Vietnamese và 100+ ngôn ngữ khác
    Model: BAAI/bge-m3
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        batch_size: int = 32,
        max_length: int = 8192,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False
    ):
        """
        Initialize BGE-M3 model
        
        Args:
            model_name: Tên model trên HuggingFace (mặc định: BAAI/bge-m3)
            device: Device để chạy model ('cuda', 'cpu', hoặc None để auto-detect)
            normalize_embeddings: Normalize embeddings về unit vectors
            use_fp16: Sử dụng FP16 để tăng tốc (chỉ với GPU)
            batch_size: Batch size khi encode
            max_length: Độ dài tối đa của input (bge-m3 support tới 8192)
            return_dense: Trả về dense embeddings (mặc định)
            return_sparse: Trả về sparse embeddings (cho BM25-like retrieval)
            return_colbert_vecs: Trả về ColBERT vectors (multi-vector retrieval)
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_length = max_length
        self.return_dense = return_dense
        self.return_sparse = return_sparse
        self.return_colbert_vecs = return_colbert_vecs
        
        # Auto-detect device nếu không được chỉ định
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load BGE-M3 model từ FlagEmbedding"""
        try:
            from FlagEmbedding import BGEM3FlagModel
            
            print(f"Loading BGE-M3 model: {self.model_name}")
            print(f"Device: {self.device}, FP16: {self.use_fp16}")
            
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device
            )
            
            print("Model loaded successfully!")
            
        except ImportError:
            raise ImportError(
                "FlagEmbedding không được cài đặt. "
                "Cài đặt bằng: pip install FlagEmbedding"
            )
        except Exception as e:
            raise RuntimeError(f"Lỗi khi load model: {str(e)}")
    
    def embed_documents(
        self, 
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> Union[List[List[float]], Dict]:
        """
        Embed danh sách documents
        
        Args:
            texts: Danh sách texts cần embed
            batch_size: Batch size (None để dùng giá trị mặc định)
            
        Returns:
            List[List[float]] hoặc Dict với dense/sparse/colbert embeddings
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.batch_size
        
        # Encode với BGE-M3
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=self.return_dense,
            return_sparse=self.return_sparse,
            return_colbert_vecs=self.return_colbert_vecs
        )
        
        # Nếu chỉ trả về dense embeddings (default)
        if self.return_dense and not self.return_sparse and not self.return_colbert_vecs:
            dense_vecs: np.ndarray = embeddings['dense_vecs']
            
            if self.normalize_embeddings:
                dense_vecs = self._normalize_embeddings(dense_vecs)
            
            return dense_vecs.tolist()
        
        # Trả về full dict nếu có sparse hoặc colbert
        result = {}
        if self.return_dense:
            dense_vecs: np.ndarray = embeddings['dense_vecs']
            if self.normalize_embeddings:
                dense_vecs = self._normalize_embeddings(dense_vecs)
            result['dense'] = dense_vecs.tolist()
        
        if self.return_sparse:
            result['sparse'] = embeddings['lexical_weights']
        
        if self.return_colbert_vecs:
            result['colbert'] = embeddings['colbert_vecs']
        
        return result
    
    def embed_query(
        self, 
        text: str
    ) -> Union[List[float], Dict]:
        """
        Embed một query
        
        Args:
            text: Query text
            
        Returns:
            List[float] hoặc Dict với embeddings
        """
        result = self.embed_documents([text], batch_size=1)
        
        if isinstance(result, list):
            return result[0]
        else:
            # Dict với multiple embedding types
            return {k: v[0] if isinstance(v, list) else v for k, v in result.items()}
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings thành unit vectors"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms
    
    @property
    def dimension(self) -> int:
        """Dimension của dense embeddings (BGE-M3: 1024)"""
        return 1024
    
    def compute_similarity(
        self,
        query_embedding: List[float],
        doc_embeddings: List[List[float]]
    ) -> List[float]:
        """
        Tính similarity scores giữa query và documents
        
        Args:
            query_embedding: Query embedding
            doc_embeddings: Danh sách document embeddings
            
        Returns:
            List[float]: Similarity scores (cosine similarity)
        """
        query_vec = np.array(query_embedding)
        doc_vecs = np.array(doc_embeddings)
        
        # Cosine similarity (đã normalized nên chỉ cần dot product)
        if self.normalize_embeddings:
            similarities = np.dot(doc_vecs, query_vec)
        else:
            # Compute cosine similarity
            query_norm = np.linalg.norm(query_vec)
            doc_norms = np.linalg.norm(doc_vecs, axis=1)
            similarities = np.dot(doc_vecs, query_vec) / (doc_norms * query_norm)
        
        return similarities.tolist()


class HuggingFaceEmbedding(BaseEmbedding):
    """
    Generic HuggingFace Embedding model
    Có thể sử dụng với bất kỳ model nào trên HuggingFace
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        max_length: int = 512
    ):
        """
        Initialize HuggingFace embedding model
        
        Args:
            model_name: Tên model trên HuggingFace
            device: Device để chạy
            normalize_embeddings: Normalize embeddings
            batch_size: Batch size
            max_length: Max sequence length
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_length = max_length
        
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self._load_model()
    
    def _load_model(self):
        """Load model từ sentence-transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded! Dimension: {self._dimension}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers không được cài đặt. "
                "Cài đặt bằng: pip install sentence-transformers"
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents"""
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embedding.tolist()
    
    @property
    def dimension(self) -> int:
        """Dimension của embeddings"""
        return self._dimension
