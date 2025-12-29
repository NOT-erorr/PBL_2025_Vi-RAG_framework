from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from pathlib import Path
import pickle
import logging
import json
from ..core.vectorstore import VectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """
    FAISS Vector Store Implementation - Tối ưu cho tiếng Việt
    
    Hỗ trợ nhiều loại FAISS index với cấu hình linh hoạt:
    - Flat: Exact search, độ chính xác cao nhất
    - IVF: Inverted File Index, tìm kiếm nhanh cho dataset lớn
    - HNSW: Hierarchical NSW, cân bằng tốt giữa tốc độ và chất lượng (recommended)
    - IVF_PQ: IVF + Product Quantization, tiết kiệm bộ nhớ
    - IVFPQ_Fast: IVF_PQ với cấu hình tối ưu cho tốc độ
    
    Tối ưu cho tiếng Việt:
    - Sử dụng cosine similarity (phù hợp với semantic embeddings)
    - Normalize vectors để cải thiện độ chính xác
    - Auto-tuning parameters dựa trên kích thước dataset
    - Batch processing hiệu quả
    """
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "HNSW",
        metric: str = "cosine",
        normalize: bool = True,
        # IVF parameters
        nlist: int = None,
        nprobe: int = None,
        # HNSW parameters
        hnsw_m: int = 32,
        hnsw_ef_search: int = 128,
        hnsw_ef_construction: int = 200,
        # PQ parameters
        pq_m: int = None,
        pq_bits: int = 8,
        # GPU parameters
        use_gpu: bool = False,
        gpu_id: int = 0,
        # Optimization
        optimize_for_vietnamese: bool = True
    ):
        """
        Khởi tạo FAISS Vector Store với cấu hình tối ưu cho tiếng Việt
        
        Args:
            dimension: Số chiều của embeddings (768 cho PhoBERT, 384 cho MiniLM)
            index_type: Loại FAISS index:
                - 'Flat': Exact search, chính xác nhất (nhỏ < 10K docs)
                - 'IVF': Nhanh hơn, cho dataset lớn (> 10K docs)
                - 'HNSW': Tốt nhất cho tiếng Việt, nhanh + chính xác (recommended)
                - 'IVF_PQ': Tiết kiệm bộ nhớ (> 100K docs)
                - 'IVFPQ_Fast': Cực nhanh, độ chính xác chấp nhận được
            metric: Distance metric:
                - 'cosine': Cosine similarity (recommended cho tiếng Việt)
                - 'l2': Euclidean distance
                - 'inner_product': Inner product
            normalize: Normalize vectors (recommended=True cho tiếng Việt)
            nlist: Số clusters cho IVF (auto: sqrt(n) hoặc 100)
            nprobe: Số clusters search trong IVF (auto: nlist/10 hoặc 10)
            hnsw_m: Connections per layer (16-64, default=32 cho tiếng Việt)
            hnsw_ef_search: Search depth (64-256, default=128 cho tiếng Việt)
            hnsw_ef_construction: Build depth (100-400, default=200)
            pq_m: Sub-quantizers cho PQ (auto: dimension/16)
            pq_bits: Bits per sub-quantizer (4-8, default=8)
            use_gpu: Sử dụng GPU (requires faiss-gpu)
            gpu_id: GPU device ID
            optimize_for_vietnamese: Tự động tối ưu cho tiếng Việt
        """
        # Import FAISS
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError(
                "FAISS chưa được cài đặt. Vui lòng cài đặt:\n"
                "  - CPU: pip install faiss-cpu\n"
                "  - GPU: pip install faiss-gpu"
            )
        
        # Store configuration
        self.dimension = dimension
        self.index_type = index_type.upper()
        self.metric = metric.lower()
        self.normalize = normalize
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        # Auto-optimize for Vietnamese if enabled
        if optimize_for_vietnamese:
            logger.info("Áp dụng tối ưu hóa cho tiếng Việt")
            if self.metric != "cosine":
                logger.warning(f"Metric '{self.metric}' không được khuyến nghị cho tiếng Việt. Đề xuất: 'cosine'")
            if not self.normalize and self.metric == "cosine":
                logger.info("Bật normalize cho cosine similarity")
                self.normalize = True
        
        # IVF parameters with auto-tuning
        self.nlist = nlist or 100
        self.nprobe = nprobe or max(1, self.nlist // 10)
        
        # HNSW parameters optimized for Vietnamese
        self.hnsw_m = hnsw_m
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_ef_construction = hnsw_ef_construction
        
        # PQ parameters with auto-tuning
        self.pq_m = pq_m or max(1, dimension // 16)
        self.pq_bits = pq_bits
        
        # Initialize index
        self.index = self._create_index()
        
        # Move to GPU if requested
        if self.use_gpu:
            self._move_to_gpu()
        
        # Storage
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []
        self.ids: List[str] = []
        self._id_to_index: Dict[str, int] = {}
        self._counter = 0
        
        logger.info(
            f"✓ Khởi tạo FAISS VectorStore thành công:\n"
            f"  - Dimension: {dimension}\n"
            f"  - Index type: {self.index_type}\n"
            f"  - Metric: {self.metric}\n"
            f"  - Normalize: {self.normalize}\n"
            f"  - GPU: {self.use_gpu}"
        )
    
    def _create_index(self):
        """
        Tạo FAISS index tối ưu cho tiếng Việt
        
        Lựa chọn index dựa trên kích thước dataset và yêu cầu
        """
        use_inner_product = self.metric in ["cosine", "inner_product"]
        
        if self.index_type == "FLAT":
            # Exact search - Tốt nhất cho dataset nhỏ (< 10K)
            if use_inner_product:
                index = self.faiss.IndexFlatIP(self.dimension)
            else:
                index = self.faiss.IndexFlatL2(self.dimension)
            logger.info("✓ Tạo Flat index - Exact search (chậm nhưng chính xác)")
        
        elif self.index_type == "IVF":
            # IVF - Tốt cho dataset trung bình (10K-100K)
            if use_inner_product:
                quantizer = self.faiss.IndexFlatIP(self.dimension)
                index = self.faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            else:
                quantizer = self.faiss.IndexFlatL2(self.dimension)
                index = self.faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            index.nprobe = self.nprobe
            logger.info(f"✓ Tạo IVF index - nlist={self.nlist}, nprobe={self.nprobe}")
        
        elif self.index_type == "HNSW":
            # HNSW - Tốt nhất cho tiếng Việt (cân bằng tốc độ + chất lượng)
            index = self.faiss.IndexHNSWFlat(self.dimension, self.hnsw_m)
            index.hnsw.efConstruction = self.hnsw_ef_construction
            index.hnsw.efSearch = self.hnsw_ef_search
            logger.info(
                f"✓ Tạo HNSW index (recommended cho tiếng Việt):\n"
                f"  - M: {self.hnsw_m}\n"
                f"  - efConstruction: {self.hnsw_ef_construction}\n"
                f"  - efSearch: {self.hnsw_ef_search}"
            )
        
        elif self.index_type == "IVF_PQ" or self.index_type == "IVFPQ":
            # IVF + PQ - Tiết kiệm bộ nhớ cho dataset lớn (> 100K)
            if use_inner_product:
                quantizer = self.faiss.IndexFlatIP(self.dimension)
                index = self.faiss.IndexIVFPQ(
                    quantizer, self.dimension, self.nlist, self.pq_m, self.pq_bits
                )
            else:
                quantizer = self.faiss.IndexFlatL2(self.dimension)
                index = self.faiss.IndexIVFPQ(
                    quantizer, self.dimension, self.nlist, self.pq_m, self.pq_bits
                )
            index.nprobe = self.nprobe
            logger.info(
                f"✓ Tạo IVF_PQ index (memory efficient):\n"
                f"  - nlist: {self.nlist}, nprobe: {self.nprobe}\n"
                f"  - PQ: m={self.pq_m}, bits={self.pq_bits}"
            )
        
        elif self.index_type == "IVFPQ_FAST":
            # Tối ưu cho tốc độ với PQ aggressive
            nlist_fast = max(64, int(np.sqrt(10000)))
            if use_inner_product:
                quantizer = self.faiss.IndexFlatIP(self.dimension)
                index = self.faiss.IndexIVFPQ(
                    quantizer, self.dimension, nlist_fast, 
                    self.dimension // 8, 4
                )
            else:
                quantizer = self.faiss.IndexFlatL2(self.dimension)
                index = self.faiss.IndexIVFPQ(
                    quantizer, self.dimension, nlist_fast,
                    self.dimension // 8, 4
                )
            index.nprobe = max(1, nlist_fast // 20)
            logger.info("✓ Tạo IVFPQ_Fast index - Tối ưu tốc độ")
        
        else:
            raise ValueError(
                f"Index type '{self.index_type}' không được hỗ trợ.\n"
                f"Các loại hỗ trợ: Flat, IVF, HNSW, IVF_PQ, IVFPQ_Fast"
            )
        
        return index
    
    def _move_to_gpu(self):
        """Di chuyển FAISS index lên GPU"""
        try:
            if not hasattr(self.faiss, 'StandardGpuResources'):
                logger.warning("GPU không khả dụng. Cài faiss-gpu để dùng GPU.")
                self.use_gpu = False
                return
            
            res = self.faiss.StandardGpuResources()
            self.index = self.faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
            logger.info(f"✓ Đã chuyển index lên GPU {self.gpu_id}")
        except Exception as e:
            logger.warning(f"Không thể chuyển lên GPU: {e}. Sử dụng CPU.")
            self.use_gpu = False
    
    def _generate_id(self) -> str:
        """Tạo ID duy nhất"""
        self._counter += 1
        return f"doc_{self._counter:08d}"
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors thành unit vectors
        Quan trọng cho cosine similarity với tiếng Việt
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Tránh chia cho 0
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def _auto_tune_ivf_params(self, num_vectors: int):
        """
        Tự động điều chỉnh IVF parameters dựa trên kích thước dataset
        Tối ưu cho tiếng Việt
        """
        if self.index_type in ["IVF", "IVF_PQ", "IVFPQ"]:
            # Auto-tune nlist based on dataset size
            if num_vectors < 1000:
                optimal_nlist = max(10, int(np.sqrt(num_vectors)))
            elif num_vectors < 10000:
                optimal_nlist = max(50, int(np.sqrt(num_vectors)))
            elif num_vectors < 100000:
                optimal_nlist = max(100, int(np.sqrt(num_vectors)))
            else:
                optimal_nlist = max(256, int(4 * np.sqrt(num_vectors)))
            
            # Update nlist if different
            if optimal_nlist != self.nlist:
                old_nlist = self.nlist
                self.nlist = optimal_nlist
                self.nprobe = max(1, min(self.nlist // 10, 50))
                logger.info(
                    f"Auto-tune IVF: nlist {old_nlist}→{self.nlist}, "
                    f"nprobe→{self.nprobe} (n={num_vectors})"
                )
                # Recreate index with new params
                self.index = self._create_index()
                if self.use_gpu:
                    self._move_to_gpu()
    
    def add_texts(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 1000
    ) -> List[str]:
        """
        Thêm texts và embeddings vào vector store
        
        Args:
            texts: Danh sách văn bản tiếng Việt
            embeddings: Embeddings tương ứng
            metadatas: Metadata (optional)
            ids: Document IDs (optional, auto-generate nếu không có)
            batch_size: Kích thước batch để xử lý
            
        Returns:
            List[str]: IDs của documents đã thêm
        """
        if not texts:
            logger.warning("Không có texts để thêm")
            return []
        
        if len(texts) != len(embeddings):
            raise ValueError(
                f"Số lượng texts ({len(texts)}) và embeddings ({len(embeddings)}) không khớp"
            )
        
        # Generate IDs if not provided
        if ids is None:
            ids = [self._generate_id() for _ in range(len(texts))]
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        elif len(metadatas) != len(texts):
            raise ValueError(f"Số lượng metadatas phải bằng số texts")
        
        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Normalize if needed (recommended for Vietnamese)
        if self.normalize:
            vectors = self._normalize_vectors(vectors)
        
        # Auto-tune IVF parameters before training
        if self.index.ntotal == 0:  # First batch
            total_expected = len(vectors)
            self._auto_tune_ivf_params(total_expected)
        
        # Train index if needed
        if self.index_type in ["IVF", "IVF_PQ", "IVFPQ", "IVFPQ_FAST"]:
            if not self.index.is_trained:
                logger.info(f"Training {self.index_type} index với {len(vectors)} vectors...")
                # Use subset for large datasets
                train_vectors = vectors if len(vectors) <= 100000 else vectors[:100000]
                self.index.train(train_vectors)
                logger.info("✓ Training hoàn tất")
        
        # Add vectors in batches
        added_ids = []
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            
            batch_vectors = vectors[i:batch_end]
            batch_texts = texts[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            # Current index position
            start_idx = len(self.texts)
            
            # Add to FAISS index
            self.index.add(batch_vectors)
            
            # Store metadata
            self.texts.extend(batch_texts)
            self.metadatas.extend(batch_metadatas)
            self.ids.extend(batch_ids)
            
            # Update ID mapping
            for j, doc_id in enumerate(batch_ids):
                self._id_to_index[doc_id] = start_idx + j
            
            added_ids.extend(batch_ids)
            
            if batch_end % batch_size == 0:
                logger.debug(f"Đã thêm {batch_end}/{len(texts)} vectors")
        
        logger.info(f"✓ Đã thêm {len(texts)} documents. Tổng: {self.index.ntotal}")
        return added_ids
    
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
    
    def search(
        self,
        query: Union[str, List[float]],
        k: int = 4,
        filter: Optional[Dict] = None,
        embedding_function = None,
        **kwargs
    ) -> List[Tuple[str, float, Dict]]:
        """
        Tìm kiếm tổng quát - hỗ trợ cả text và embedding
        
        Args:
            query: Câu hỏi tiếng Việt hoặc embedding vector
            k: Số lượng kết quả
            filter: Lọc metadata
            embedding_function: Function để embed text (required nếu query là string)
            
        Returns:
            List[Tuple[str, float, Dict]]: Kết quả tìm kiếm
        """
        # Convert text to embedding if needed
        if isinstance(query, str):
            if embedding_function is None:
                raise ValueError(
                    "embedding_function bắt buộc khi query là text.\n"
                    "Truyền embedding_function hoặc dùng similarity_search với embedding."
                )
            query_embedding = embedding_function([query])[0]
        else:
            query_embedding = query
        
        return self.similarity_search(query_embedding, k=k, filter=filter)
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Tìm kiếm văn bản tương tự bằng embedding
        
        Args:
            query_embedding: Query embedding vector
            k: Số lượng kết quả
            filter: Lọc metadata (optional)
            
        Returns:
            List[Tuple[str, float, Dict]]: (text, score, metadata)
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store trống")
            return []
        
        # Convert to numpy
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Normalize if needed
        if self.normalize:
            query_vector = self._normalize_vectors(query_vector)
        
        # Search with filter buffer
        search_k = min(k * 3 if filter else k, self.index.ntotal)
        
        try:
            # FAISS search returns (distances, indices) tuple
            distances, indices = self.index.search(query_vector, search_k)  # type: ignore
        except Exception as e:
            logger.error(f"Lỗi khi search: {e}")
            return []
        
        # Process results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.texts):
                continue
            
            idx = int(idx)
            text = self.texts[idx]
            metadata = self.metadatas[idx]
            
            # Apply filter
            if filter:
                if not all(metadata.get(k) == v for k, v in filter.items()):
                    continue
            
            # Convert distance to score
            if self.metric in ["cosine", "inner_product"]:
                score = float(dist)  # Higher is better
            else:  # l2
                score = float(-dist)  # Lower distance = higher score
            
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
        distances, indices = self.index.search(query_vector, search_k)  # type: ignore
        
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
        
        Note: FAISS không hỗ trợ xóa trực tiếp, cần rebuild index
        
        Args:
            ids: Danh sách IDs cần xóa
            
        Returns:
            bool: True nếu thành công
        """
        if not ids:
            return True
        
        # Find indices to remove
        indices_to_remove = set()
        for doc_id in ids:
            if doc_id in self._id_to_index:
                indices_to_remove.add(self._id_to_index[doc_id])
        
        if not indices_to_remove:
            logger.warning(f"Không tìm thấy IDs để xóa: {ids}")
            return True
        
        logger.info(f"Đang xóa {len(indices_to_remove)} documents...")
        
        # Collect remaining data
        remaining_texts = []
        remaining_metadatas = []
        remaining_ids = []
        remaining_embeddings = []
        
        for i in range(len(self.texts)):
            if i not in indices_to_remove:
                remaining_texts.append(self.texts[i])
                remaining_metadatas.append(self.metadatas[i])
                remaining_ids.append(self.ids[i])
                # Reconstruct embedding
                try:
                    vec = self.index.reconstruct(i)
                    remaining_embeddings.append(vec.tolist())
                except:
                    logger.warning(f"Không thể reconstruct vector tại index {i}")
        
        # Rebuild index
        self.clear()
        
        if remaining_texts:
            self.add_texts(
                texts=remaining_texts,
                embeddings=remaining_embeddings,
                metadatas=remaining_metadatas,
                ids=remaining_ids
            )
        
        logger.info(f"✓ Đã xóa {len(indices_to_remove)} documents")
        return True
    
    def save(self, path: str) -> None:
        """
        Lưu vector store vào disk
        
        Args:
            path: Đường dẫn thư mục để lưu
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = save_path / "faiss.index"
        if self.use_gpu:
            # Move to CPU before saving
            cpu_index = self.faiss.index_gpu_to_cpu(self.index)
            self.faiss.write_index(cpu_index, str(index_file))
        else:
            self.faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        metadata = {
            "texts": self.texts,
            "metadatas": self.metadatas,
            "ids": self.ids,
            "id_to_index": self._id_to_index,
            "counter": self._counter,
            "config": {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
                "normalize": self.normalize,
                "nlist": self.nlist,
                "nprobe": self.nprobe,
                "hnsw_m": self.hnsw_m,
                "hnsw_ef_search": self.hnsw_ef_search,
                "hnsw_ef_construction": self.hnsw_ef_construction,
                "pq_m": self.pq_m,
                "pq_bits": self.pq_bits,
            }
        }
        
        metadata_file = save_path / "metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save config as JSON for readability
        config_file = save_path / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(metadata["config"], f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Đã lưu vector store vào: {path}")
        logger.info(f"  - Index: {index_file}")
        logger.info(f"  - Metadata: {metadata_file}")
        logger.info(f"  - Tổng vectors: {self.index.ntotal}")
    
    def load(self, path: str) -> None:
        """
        Load vector store từ disk
        
        Args:
            path: Đường dẫn thư mục đã lưu
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Không tìm thấy đường dẫn: {path}")
        
        # Load FAISS index
        index_file = load_path / "faiss.index"
        if not index_file.exists():
            # Try old naming
            index_file = load_path / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"Không tìm thấy file index: {index_file}")
        
        self.index = self.faiss.read_index(str(index_file))
        
        # Move to GPU if needed
        if self.use_gpu:
            self._move_to_gpu()
        
        # Load metadata
        metadata_file = load_path / "metadata.pkl"
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        
        self.texts = metadata["texts"]
        self.metadatas = metadata["metadatas"]
        self.ids = metadata["ids"]
        self._id_to_index = metadata["id_to_index"]
        self._counter = metadata["counter"]
        
        # Restore config
        config = metadata.get("config", {})
        self.dimension = config.get("dimension", metadata.get("dimension", self.dimension))
        self.index_type = config.get("index_type", metadata.get("index_type", self.index_type))
        self.metric = config.get("metric", metadata.get("metric", self.metric))
        self.normalize = config.get("normalize", metadata.get("normalize", self.normalize))
        self.nlist = config.get("nlist", 100)
        self.nprobe = config.get("nprobe", 10)
        self.hnsw_m = config.get("hnsw_m", 32)
        self.hnsw_ef_search = config.get("hnsw_ef_search", 128)
        self.hnsw_ef_construction = config.get("hnsw_ef_construction", 200)
        self.pq_m = config.get("pq_m", self.dimension // 16)
        self.pq_bits = config.get("pq_bits", 8)
        
        logger.info(f"✓ Đã load vector store từ: {path}")
        logger.info(f"  - Tổng vectors: {self.index.ntotal}")
        logger.info(f"  - Index type: {self.index_type}")
        logger.info(f"  - Metric: {self.metric}")
    
    def get_count(self) -> int:
        """
        Lấy số lượng vectors trong store
        
        Returns:
            int: Số lượng vectors
        """
        return self.index.ntotal
    
    def clear(self) -> None:
        """
        Xóa tất cả dữ liệu trong vector store
        """
        # Recreate index
        self.index = self._create_index()
        
        # Move to GPU if needed
        if self.use_gpu:
            self._move_to_gpu()
        
        # Clear storage
        self.texts = []
        self.metadatas = []
        self.ids = []
        self._id_to_index = {}
        self._counter = 0
        
        logger.info("✓ Đã xóa toàn bộ vector store")
    
    # === Additional utility methods ===
    
    def get_by_ids(self, ids: List[str]) -> List[Tuple[str, Dict]]:
        """
        Lấy documents theo IDs
        
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
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về index
        
        Returns:
            Dict: Statistics về vector store
        """
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "normalize": self.normalize,
            "is_trained": getattr(self.index, "is_trained", True),
            "use_gpu": self.use_gpu,
        }
        
        # Add index-specific stats
        if self.index_type in ["IVF", "IVF_PQ", "IVFPQ", "IVFPQ_FAST"]:
            stats["nlist"] = self.nlist
            stats["nprobe"] = self.nprobe
        
        if self.index_type == "HNSW":
            stats["hnsw_m"] = self.hnsw_m
            stats["hnsw_ef_search"] = self.hnsw_ef_search
            stats["hnsw_ef_construction"] = self.hnsw_ef_construction
        
        if self.index_type in ["IVF_PQ", "IVFPQ", "IVFPQ_FAST"]:
            stats["pq_m"] = self.pq_m
            stats["pq_bits"] = self.pq_bits
        
        return stats
    
    def update_search_params(self, **kwargs):
        """
        Cập nhật search parameters động
        
        Args:
            nprobe: Số clusters search trong IVF
            efSearch: Search depth cho HNSW
        """
        updated = []
        
        if "nprobe" in kwargs and hasattr(self.index, "nprobe"):
            self.index.nprobe = kwargs["nprobe"]
            self.nprobe = kwargs["nprobe"]
            updated.append(f"nprobe={kwargs['nprobe']}")
        
        if "efSearch" in kwargs and hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = kwargs["efSearch"]
            self.hnsw_ef_search = kwargs["efSearch"]
            updated.append(f"efSearch={kwargs['efSearch']}")
        
        if updated:
            logger.info(f"✓ Cập nhật search params: {', '.join(updated)}")
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding_function,
        metadatas: Optional[List[Dict]] = None,
        dimension: int = 768,
        index_type: str = "HNSW",
        batch_size: int = 1000,
        **kwargs
    ) -> "FAISSVectorStore":
        """
        Tạo vector store từ texts và embedding function
        
        Args:
            texts: Danh sách văn bản tiếng Việt
            embedding_function: Function để tạo embeddings
            metadatas: Metadata (optional)
            dimension: Embedding dimension
            index_type: Loại FAISS index
            batch_size: Batch size
            **kwargs: Tham số khác cho FAISSVectorStore
            
        Returns:
            FAISSVectorStore: Vector store đã được populated
        """
        # Create vector store
        vectorstore = cls(
            dimension=dimension,
            index_type=index_type,
            **kwargs
        )
        
        if not texts:
            logger.warning("Không có texts để thêm")
            return vectorstore
        
        logger.info(f"Đang tạo embeddings cho {len(texts)} văn bản tiếng Việt...")
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embedding_function(batch_texts)
            all_embeddings.extend(batch_embeddings)
            logger.info(
                f"  Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}: "
                f"{len(batch_embeddings)} embeddings"
            )
        
        # Add to vector store
        vectorstore.add_texts(
            texts=texts,
            embeddings=all_embeddings,
            metadatas=metadatas,
            batch_size=batch_size
        )
        
        logger.info(f"✓ Tạo vector store thành công với {vectorstore.get_count()} vectors")
        return vectorstore
    
    def __repr__(self) -> str:
        return (
            f"FAISSVectorStore(vectors={self.index.ntotal}, "
            f"type={self.index_type}, metric={self.metric}, "
            f"dim={self.dimension})"
        )
