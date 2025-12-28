from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from pathlib import Path
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        dimension: int = 768,
        index_type: str = "Flat",
        metric: str = "cosine",
        normalize: bool = True,
        nlist: int = 100,
        nprobe: int = 10,
        hnsw_m: int = 32,
        hnsw_ef_search: int = 64,
        hnsw_ef_construction: int = 200,
        use_gpu: bool = False,
        gpu_id: int = 0
    ):
        """
        Initialize FAISS Vector Store với flexible configuration
        
        Args:
            dimension: Dimension của embeddings (default: 768 cho sentence-transformers)
            index_type: Loại FAISS index:
                - 'Flat': Exact search, best quality
                - 'IVF': Inverted File Index, faster search
                - 'HNSW': Hierarchical NSW, best speed/quality tradeoff
                - 'IVF_PQ': IVF + Product Quantization, memory efficient
                - 'LSH': Locality Sensitive Hashing
            metric: Distance metric ('cosine', 'l2', 'inner_product')
            normalize: Normalize vectors trước khi add (recommended cho cosine)
            nlist: Số clusters cho IVF (default: 100)
            nprobe: Số clusters để search trong IVF (default: 10)
            hnsw_m: Số connections per layer cho HNSW (default: 32)
            hnsw_ef_search: Search depth cho HNSW (default: 64)
            hnsw_ef_construction: Construction depth cho HNSW (default: 200)
            use_gpu: Sử dụng GPU nếu có (requires faiss-gpu)
            gpu_id: GPU ID để sử dụng (default: 0)
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
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_ef_construction = hnsw_ef_construction
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Apply GPU if requested
        if self.use_gpu:
            self._move_to_gpu()
        
        # Storage cho texts và metadata
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []
        self.ids: List[str] = []
        self._id_to_index: Dict[str, int] = {}
        self._counter = 0
        
        logger.info(f"Initialized FAISS VectorStore: dim={dimension}, type={index_type}, metric={metric}")
    
    def _create_index(self):
        """Tạo FAISS index theo config với nhiều options"""
        # Determine base metric
        use_inner_product = self.metric in ["cosine", "inner_product"]
        
        if self.index_type == "Flat":
            # Exact search, best quality
            if use_inner_product:
                index = self.faiss.IndexFlatIP(self.dimension)
            else:
                index = self.faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == "IVF":
            # Inverted File Index
            if use_inner_product:
                quantizer = self.faiss.IndexFlatIP(self.dimension)
                index = self.faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            else:
                quantizer = self.faiss.IndexFlatL2(self.dimension)
                index = self.faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            index.nprobe = self.nprobe
        
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World
            index = self.faiss.IndexHNSWFlat(self.dimension, self.hnsw_m)
            index.hnsw.efConstruction = self.hnsw_ef_construction
            index.hnsw.efSearch = self.hnsw_ef_search
        
        elif self.index_type == "IVF_PQ":
            # IVF + Product Quantization (memory efficient)
            # Use 8 bytes per vector (very compressed)
            m = 8  # Number of sub-quantizers
            bits = 8  # Bits per sub-quantizer
            
            if use_inner_product:
                quantizer = self.faiss.IndexFlatIP(self.dimension)
                index = self.faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, m, bits)
            else:
                quantizer = self.faiss.IndexFlatL2(self.dimension)
                index = self.faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, m, bits)
            index.nprobe = self.nprobe
        
        elif self.index_type == "LSH":
            # Locality Sensitive Hashing
            nbits = self.dimension * 4  # Number of bits in hash
            index = self.faiss.IndexLSH(self.dimension, nbits)
        
        else:
            raise ValueError(
                f"Unsupported index_type: {self.index_type}. "
                f"Supported types: Flat, IVF, HNSW, IVF_PQ, LSH"
            )
        
        logger.info(f"Created FAISS index: {self.index_type} with metric {self.metric}")
        return index
    
    def _move_to_gpu(self):
        """Move FAISS index to GPU"""
        try:
            if not hasattr(self.faiss, 'StandardGpuResources'):
                logger.warning("GPU not available. Install faiss-gpu for GPU support.")
                self.use_gpu = False
                return
            
            res = self.faiss.StandardGpuResources()
            self.index = self.faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
            logger.info(f"Moved FAISS index to GPU {self.gpu_id}")
        except Exception as e:
            logger.warning(f"Failed to move to GPU: {e}. Using CPU instead.")
            self.use_gpu = False
    
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
        ids: Optional[List[str]] = None,
        batch_size: int = 1000
    ) -> List[str]:
        """Thêm texts và embeddings vào FAISS index với batch processing"""
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
        
        # Train index nếu cần
        if self.index_type in ["IVF", "IVF_PQ"] and not self.index.is_trained:
            logger.info(f"Training {self.index_type} index with {len(vectors)} vectors...")
            # Use a sample for training if too many vectors
            training_vectors = vectors if len(vectors) <= 100000 else vectors[:100000]
            self.index.train(training_vectors)
            logger.info("Index training completed")
        
        # Add vectors in batches
        all_ids = []
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            
            batch_vectors = vectors[i:batch_end]
            batch_texts = texts[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            # Lấy index hiện tại
            start_idx = len(self.texts)
            
            # Add vectors to FAISS index
            self.index.add(batch_vectors)
            
            # Store texts, metadata, và IDs
            self.texts.extend(batch_texts)
            self.metadatas.extend(batch_metadatas)
            self.ids.extend(batch_ids)
            
            # Update ID mapping
            for j, doc_id in enumerate(batch_ids):
                self._id_to_index[doc_id] = start_idx + j
            
            all_ids.extend(batch_ids)
            
            if batch_end - i == batch_size:
                logger.debug(f"Added batch {i//batch_size + 1}: {batch_end}/{len(texts)} vectors")
        
        logger.info(f"Added {len(texts)} vectors to index. Total: {self.index.ntotal}")
        return all_ids
    
    def add_documents(
        self,
        documents: List[Any],
        embeddings: List[List[float]],
        batch_size: int = 1000
    ) -> List[str]:
        """
        Thêm documents từ loader.py (Document objects) vào vector store
        
        Args:
            documents: List of Document objects from loader.py
            embeddings: Embeddings tương ứng với documents
            batch_size: Batch size cho processing
            
        Returns:
            List[str]: IDs của documents đã thêm
        """
        if not documents:
            return []
        
        if len(documents) != len(embeddings):
            raise ValueError("Số lượng documents và embeddings phải bằng nhau")
        
        # Extract texts và metadatas từ Document objects
        texts = []
        metadatas = []
        
        for doc in documents:
            # Assume Document has .content và .metadata attributes
            if hasattr(doc, 'content'):
                texts.append(doc.content)
            elif hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            else:
                texts.append(str(doc))
            
            if hasattr(doc, 'metadata'):
                metadatas.append(doc.metadata)
            else:
                metadatas.append({})
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        return self.add_texts(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            batch_size=batch_size
        )
    
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
        
        # Apply GPU if needed
        if self.use_gpu:
            self._move_to_gpu()
        
        self.texts = []
        self.metadatas = []
        self.ids = []
        self._id_to_index = {}
        self._counter = 0
        logger.info("Vector store cleared")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Lấy thông tin thống kê về index
        
        Returns:
            Dict: Statistics về index
        """
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "normalize": self.normalize,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            "use_gpu": self.use_gpu
        }
        
        # Add index-specific stats
        if self.index_type == "IVF" or self.index_type == "IVF_PQ":
            stats["nlist"] = self.nlist
            stats["nprobe"] = self.nprobe
        elif self.index_type == "HNSW":
            stats["hnsw_m"] = self.hnsw_m
            stats["hnsw_ef_search"] = self.hnsw_ef_search
        
        return stats
    
    def update_search_params(self, **kwargs):
        """
        Update search parameters dynamically
        
        Args:
            **kwargs: Parameters to update (e.g., nprobe, efSearch)
        """
        if "nprobe" in kwargs and hasattr(self.index, "nprobe"):
            self.index.nprobe = kwargs["nprobe"]
            self.nprobe = kwargs["nprobe"]
            logger.info(f"Updated nprobe to {kwargs['nprobe']}")
        
        if "efSearch" in kwargs and hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = kwargs["efSearch"]
            self.hnsw_ef_search = kwargs["efSearch"]
            logger.info(f"Updated efSearch to {kwargs['efSearch']}")
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Any],
        embedding_function,
        dimension: int = 768,
        index_type: str = "Flat",
        metric: str = "cosine",
        batch_size: int = 1000,
        **kwargs
    ) -> "FAISSVectorStore":
        """
        Factory method: Tạo vector store từ documents và embedding function
        
        Args:
            documents: List of Document objects
            embedding_function: Function để generate embeddings (callable)
            dimension: Embedding dimension
            index_type: FAISS index type
            metric: Distance metric
            batch_size: Batch size for processing
            **kwargs: Additional parameters cho FAISSVectorStore
            
        Returns:
            FAISSVectorStore: Vector store đã được populate
        """
        # Create vector store
        vectorstore = cls(
            dimension=dimension,
            index_type=index_type,
            metric=metric,
            **kwargs
        )
        
        if not documents:
            logger.warning("No documents provided")
            return vectorstore
        
        # Extract texts
        texts = []
        for doc in documents:
            if hasattr(doc, 'content'):
                texts.append(doc.content)
            elif hasattr(doc, 'page_content'):
                texts.append(doc.page_content)
            else:
                texts.append(str(doc))
        
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embedding_function(batch_texts)
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Add to vector store
        vectorstore.add_documents(documents, all_embeddings, batch_size=batch_size)
        
        logger.info(f"Created vector store with {vectorstore.get_count()} vectors")
        return vectorstore
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding_function,
        metadatas: Optional[List[Dict]] = None,
        dimension: int = 768,
        index_type: str = "Flat",
        metric: str = "cosine",
        batch_size: int = 1000,
        **kwargs
    ) -> "FAISSVectorStore":
        """
        Factory method: Tạo vector store từ texts và embedding function
        
        Args:
            texts: List of texts
            embedding_function: Function để generate embeddings
            metadatas: Optional metadata cho mỗi text
            dimension: Embedding dimension
            index_type: FAISS index type
            metric: Distance metric
            batch_size: Batch size for processing
            **kwargs: Additional parameters
            
        Returns:
            FAISSVectorStore: Vector store đã được populate
        """
        # Create vector store
        vectorstore = cls(
            dimension=dimension,
            index_type=index_type,
            metric=metric,
            **kwargs
        )
        
        if not texts:
            logger.warning("No texts provided")
            return vectorstore
        
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embedding_function(batch_texts)
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Add to vector store
        vectorstore.add_texts(
            texts=texts,
            embeddings=all_embeddings,
            metadatas=metadatas,
            batch_size=batch_size
        )
        
        logger.info(f"Created vector store with {vectorstore.get_count()} vectors")
        return vectorstore
