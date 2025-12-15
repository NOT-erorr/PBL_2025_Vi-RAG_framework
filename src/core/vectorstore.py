from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path
import pickle


class VectorStore(ABC):
    """Abstract base class cho vector stores"""
    
    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Thêm texts và embeddings vào vector store
        
        Args:
            texts: Danh sách texts
            embeddings: Danh sách embeddings tương ứng
            metadatas: Metadata cho mỗi text (optional)
            ids: IDs cho mỗi text (optional, sẽ auto-generate nếu không có)
            
        Returns:
            List[str]: Danh sách IDs của texts đã thêm
        """
        pass
    
    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Tìm kiếm k texts similar nhất với query
        
        Args:
            query_embedding: Query embedding vector
            k: Số lượng kết quả trả về
            filter: Filter metadata (optional)
            
        Returns:
            List[Tuple[str, float, Dict]]: List of (text, score, metadata)
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """
        Xóa texts theo IDs
        
        Args:
            ids: Danh sách IDs cần xóa
            
        Returns:
            bool: True nếu thành công
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Lưu vector store vào disk
        
        Args:
            path: Đường dẫn lưu
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load vector store từ disk
        
        Args:
            path: Đường dẫn load
        """
        pass
    
    @abstractmethod
    def get_count(self) -> int:
        """
        Lấy số lượng vectors trong store
        
        Returns:
            int: Số lượng vectors
        """
        pass


class FAISSVectorStore(VectorStore):
    """
    FAISS Vector Store implementation
    Sử dụng FAISS để lưu trữ và tìm kiếm vectors hiệu quả
    """
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "Flat",
        metric: str = "cosine",
        normalize: bool = True
    ):
        """
        Initialize FAISS Vector Store
        
        Args:
            dimension: Dimension của embeddings
            index_type: Loại FAISS index ('Flat', 'IVF', 'HNSW')
            metric: Distance metric ('cosine', 'l2', 'inner_product')
            normalize: Normalize vectors trước khi add (recommended cho cosine)
        """
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss không được cài đặt. "
                "Cài đặt bằng: pip install faiss-cpu hoặc pip install faiss-gpu"
            )
        
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.normalize = normalize
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Storage cho texts và metadata
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []
        self.ids: List[str] = []
        self._id_to_index: Dict[str, int] = {}
        self._counter = 0
    
    def _create_index(self):
        """Tạo FAISS index theo config"""
        if self.metric == "cosine" or self.metric == "inner_product":
            # Cosine similarity = Inner product với normalized vectors
            if self.index_type == "Flat":
                index = self.faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IVF":
                quantizer = self.faiss.IndexFlatIP(self.dimension)
                index = self.faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "HNSW":
                index = self.faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                raise ValueError(f"Unsupported index_type: {self.index_type}")
        
        elif self.metric == "l2":
            if self.index_type == "Flat":
                index = self.faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IVF":
                quantizer = self.faiss.IndexFlatL2(self.dimension)
                index = self.faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "HNSW":
                index = self.faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                raise ValueError(f"Unsupported index_type: {self.index_type}")
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        return index
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        self._counter += 1
        return f"doc_{self._counter}"
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors thành unit vectors"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms
    
    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Thêm texts và embeddings vào FAISS index"""
        if not texts:
            return []
        
        if len(texts) != len(embeddings):
            raise ValueError("Số lượng texts và embeddings phải bằng nhau")
        
        # Generate IDs nếu không có
        if ids is None:
            ids = [self._generate_id() for _ in range(len(texts))]
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        elif len(metadatas) != len(texts):
            raise ValueError("Số lượng metadatas phải bằng số texts")
        
        # Convert embeddings to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Normalize nếu cần
        if self.normalize:
            vectors = self._normalize_vectors(vectors)
        
        # Train index nếu là IVF và chưa được train
        if self.index_type == "IVF" and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(vectors)
        
        # Lấy index hiện tại
        start_idx = len(self.texts)
        
        # Add vectors to FAISS index
        self.index.add(vectors)
        
        # Store texts, metadata, và IDs
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # Update ID mapping
        for i, doc_id in enumerate(ids):
            self._id_to_index[doc_id] = start_idx + i
        
        return ids
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """Tìm kiếm với FAISS"""
        if self.index.ntotal == 0:
            return []
        
        # Convert query to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Normalize nếu cần
        if self.normalize:
            query_vector = self._normalize_vectors(query_vector)
        
        # Search trong FAISS
        # Lấy nhiều hơn k nếu có filter để đảm bảo đủ kết quả sau khi filter
        search_k = min(k * 3 if filter else k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, search_k)
        
        # Convert kết quả
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS trả về -1 nếu không tìm thấy đủ
                continue
            
            idx = int(idx)
            text = self.texts[idx]
            metadata = self.metadatas[idx]
            
            # Apply filter nếu có
            if filter:
                match = all(
                    metadata.get(key) == value 
                    for key, value in filter.items()
                )
                if not match:
                    continue
            
            # Convert distance to similarity score
            if self.metric == "cosine" or self.metric == "inner_product":
                score = float(dist)  # Higher is better
            else:  # l2
                score = float(-dist)  # Lower distance is better, negate for sorting
            
            results.append((text, score, metadata))
            
            if len(results) >= k:
                break
        
        return results
    
    def similarity_search_with_ids(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Tìm kiếm với IDs
        
        Returns:
            List[Tuple[str, str, float, Dict]]: (id, text, score, metadata)
        """
        if self.index.ntotal == 0:
            return []
        
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        if self.normalize:
            query_vector = self._normalize_vectors(query_vector)
        
        search_k = min(k * 3 if filter else k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            idx = int(idx)
            doc_id = self.ids[idx]
            text = self.texts[idx]
            metadata = self.metadatas[idx]
            
            if filter:
                match = all(
                    metadata.get(key) == value 
                    for key, value in filter.items()
                )
                if not match:
                    continue
            
            if self.metric == "cosine" or self.metric == "inner_product":
                score = float(dist)
            else:
                score = float(-dist)
            
            results.append((doc_id, text, score, metadata))
            
            if len(results) >= k:
                break
        
        return results
    
    def delete(self, ids: List[str]) -> bool:
        """
        Xóa documents theo IDs
        Note: FAISS không support xóa trực tiếp, cần rebuild index
        """
        if not ids:
            return True
        
        # Tìm indices cần xóa
        indices_to_remove = set()
        for doc_id in ids:
            if doc_id in self._id_to_index:
                indices_to_remove.add(self._id_to_index[doc_id])
        
        if not indices_to_remove:
            return True
        
        # Rebuild index without removed items
        remaining_texts = []
        remaining_metadatas = []
        remaining_ids = []
        remaining_embeddings = []
        
        for i in range(len(self.texts)):
            if i not in indices_to_remove:
                remaining_texts.append(self.texts[i])
                remaining_metadatas.append(self.metadatas[i])
                remaining_ids.append(self.ids[i])
                # Reconstruct embedding from FAISS
                vec = self.index.reconstruct(i)
                remaining_embeddings.append(vec.tolist())
        
        # Reset và rebuild
        self.index = self._create_index()
        self.texts = []
        self.metadatas = []
        self.ids = []
        self._id_to_index = {}
        
        if remaining_texts:
            self.add_texts(
                remaining_texts,
                remaining_embeddings,
                remaining_metadatas,
                remaining_ids
            )
        
        return True
    
    def save(self, path: str) -> None:
        """Lưu FAISS index và metadata"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "index.faiss"
        self.faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = path / "metadata.pkl"
        metadata = {
            "texts": self.texts,
            "metadatas": self.metadatas,
            "ids": self.ids,
            "id_to_index": self._id_to_index,
            "counter": self._counter,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "normalize": self.normalize
        }
        
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        print(f"Vector store saved to {path}")
    
    def load(self, path: str) -> None:
        """Load FAISS index và metadata"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Path không tồn tại: {path}")
        
        # Load FAISS index
        index_path = path / "index.faiss"
        self.index = self.faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        self.texts = metadata["texts"]
        self.metadatas = metadata["metadatas"]
        self.ids = metadata["ids"]
        self._id_to_index = metadata["id_to_index"]
        self._counter = metadata["counter"]
        self.dimension = metadata["dimension"]
        self.index_type = metadata["index_type"]
        self.metric = metadata["metric"]
        self.normalize = metadata["normalize"]
        
        print(f"Vector store loaded from {path}")
        print(f"Total vectors: {self.index.ntotal}")
    
    def get_count(self) -> int:
        """Lấy số lượng vectors"""
        return self.index.ntotal
    
    def get_by_ids(self, ids: List[str]) -> List[Tuple[str, Dict]]:
        """
        Lấy texts và metadata theo IDs
        
        Args:
            ids: Danh sách IDs
            
        Returns:
            List[Tuple[str, Dict]]: (text, metadata)
        """
        results = []
        for doc_id in ids:
            if doc_id in self._id_to_index:
                idx = self._id_to_index[doc_id]
                results.append((self.texts[idx], self.metadatas[idx]))
        return results
    
    def clear(self) -> None:
        """Xóa tất cả dữ liệu trong vector store"""
        self.index = self._create_index()
        self.texts = []
        self.metadatas = []
        self.ids = []
        self._id_to_index = {}
        self._counter = 0
        print("Vector store cleared")
