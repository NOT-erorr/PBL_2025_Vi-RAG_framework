from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass


class RetrievalStrategy(Enum):
    """Enum định nghĩa các chiến lược retrieval"""
    FLAT = "flat"  # Retrieval thông thường, không dùng hierarchy
    SMALL_TO_BIG = "small_to_big"  # Retrieve small chunks, return với parent context
    CHILD_FIRST = "child_first"  # Retrieve children trước, có thể expand sang parent
    PARENT_FIRST = "parent_first"  # Retrieve parent trước, có thể expand sang children
    MULTI_LEVEL = "multi_level"  # Retrieve ở nhiều levels đồng thời


@dataclass
class RetrievalResult:
    """
    Dataclass đại diện cho kết quả retrieval
    """
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    parent_content: Optional[str] = None
    children_contents: Optional[List[str]] = None
    level: Optional[str] = None


class BaseRetriever(ABC):
    """Abstract base class cho retrievers"""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve documents liên quan đến query
        
        Args:
            query: Query string
            k: Số lượng kết quả trả về
            **kwargs: Additional parameters
            
        Returns:
            List[RetrievalResult]: Danh sách kết quả
        """
        pass


class VectorStoreRetriever(BaseRetriever):
    """
    Basic Vector Store Retriever
    Retrieve dựa trên similarity search trong vector store
    """
    
    def __init__(self, vectorstore, embedding_model):
        """
        Args:
            vectorstore: VectorStore instance
            embedding_model: Embedding model instance
        """
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
    
    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve documents từ vector store"""
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search trong vector store
        results = self.vectorstore.similarity_search(
            query_embedding,
            k=k,
            filter=filter
        )
        
        # Convert sang RetrievalResult
        retrieval_results = []
        for text, score, metadata in results:
            result = RetrievalResult(
                content=text,
                score=score,
                metadata=metadata,
                chunk_id=metadata.get("chunk_id")
            )
            retrieval_results.append(result)
        
        return retrieval_results


class HierarchicalRetriever(BaseRetriever):
    """
    Hierarchical Retriever cho parent-child chunking
    Support nhiều retrieval strategies khác nhau
    """
    
    def __init__(
        self,
        vectorstore,
        embedding_model,
        chunk_map: Dict[str, Dict[str, Any]],
        strategy: RetrievalStrategy = RetrievalStrategy.SMALL_TO_BIG,
        child_k: int = 10,
        parent_k: int = 3
    ):
        """
        Args:
            vectorstore: VectorStore instance chứa các chunks
            embedding_model: Embedding model
            chunk_map: Dictionary mapping chunk_id -> chunk info (content, parent_id, children_ids, level)
            strategy: Retrieval strategy
            child_k: Số lượng child chunks để retrieve (cho CHILD_FIRST)
            parent_k: Số lượng parent chunks để retrieve (cho PARENT_FIRST)
        """
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.chunk_map = chunk_map
        self.strategy = strategy
        self.child_k = child_k
        self.parent_k = parent_k
    
    def retrieve(
        self,
        query: str,
        k: int = 4,
        strategy: Optional[RetrievalStrategy] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve với hierarchical strategy
        
        Args:
            query: Query string
            k: Số lượng kết quả cuối cùng
            strategy: Override default strategy (optional)
            
        Returns:
            List[RetrievalResult]: Kết quả với parent-child context
        """
        strategy = strategy or self.strategy
        
        if strategy == RetrievalStrategy.FLAT:
            return self._retrieve_flat(query, k)
        elif strategy == RetrievalStrategy.SMALL_TO_BIG:
            return self._retrieve_small_to_big(query, k)
        elif strategy == RetrievalStrategy.CHILD_FIRST:
            return self._retrieve_child_first(query, k)
        elif strategy == RetrievalStrategy.PARENT_FIRST:
            return self._retrieve_parent_first(query, k)
        elif strategy == RetrievalStrategy.MULTI_LEVEL:
            return self._retrieve_multi_level(query, k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _retrieve_flat(self, query: str, k: int) -> List[RetrievalResult]:
        """Flat retrieval không dùng hierarchy"""
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vectorstore.similarity_search_with_ids(query_embedding, k=k)
        
        retrieval_results = []
        for doc_id, text, score, metadata in results:
            result = RetrievalResult(
                content=text,
                score=score,
                metadata=metadata,
                chunk_id=doc_id,
                level=metadata.get("level")
            )
            retrieval_results.append(result)
        
        return retrieval_results
    
    def _retrieve_small_to_big(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Small-to-Big: Retrieve small chunks (sentence/paragraph level)
        nhưng return với parent context để có thêm ngữ cảnh
        
        Strategy này tốt cho:
        - Precision: Retrieve chính xác chunks nhỏ relevant
        - Context: Nhưng vẫn có ngữ cảnh rộng hơn từ parent
        """
        query_embedding = self.embedding_model.embed_query(query)
        
        # Retrieve nhiều small chunks hơn
        results = self.vectorstore.similarity_search_with_ids(
            query_embedding, 
            k=k * 2
        )
        
        retrieval_results = []
        seen_parents = set()
        
        for doc_id, text, score, metadata in results:
            chunk_info = self.chunk_map.get(doc_id)
            
            if not chunk_info:
                # Nếu không có thông tin hierarchy, return chunk thông thường
                result = RetrievalResult(
                    content=text,
                    score=score,
                    metadata=metadata,
                    chunk_id=doc_id
                )
                retrieval_results.append(result)
                continue
            
            parent_id = chunk_info.get("parent_id")
            
            # Nếu có parent và chưa thấy parent này
            if parent_id and parent_id not in seen_parents:
                parent_info = self.chunk_map.get(parent_id)
                
                if parent_info:
                    seen_parents.add(parent_id)
                    
                    # Return parent content với child được highlight
                    result = RetrievalResult(
                        content=parent_info["content"],
                        score=score,  # Giữ score của child
                        metadata={
                            **metadata,
                            "matched_child_id": doc_id,
                            "matched_child_content": text,
                            "retrieval_strategy": "small_to_big"
                        },
                        chunk_id=parent_id,
                        parent_content=parent_info.get("parent_content"),
                        level=parent_info.get("level")
                    )
                    retrieval_results.append(result)
                else:
                    # Fallback: return child nếu không tìm thấy parent
                    result = RetrievalResult(
                        content=text,
                        score=score,
                        metadata=metadata,
                        chunk_id=doc_id,
                        level=chunk_info.get("level")
                    )
                    retrieval_results.append(result)
            elif not parent_id:
                # Chunk này là top-level, return trực tiếp
                result = RetrievalResult(
                    content=text,
                    score=score,
                    metadata=metadata,
                    chunk_id=doc_id,
                    level=chunk_info.get("level")
                )
                retrieval_results.append(result)
            
            if len(retrieval_results) >= k:
                break
        
        return retrieval_results[:k]
    
    def _retrieve_child_first(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Child-First: Retrieve children chunks trước, 
        nhưng có thể expand để lấy thêm parent context
        
        Strategy này tốt cho:
        - Detailed retrieval: Tìm kiếm ở mức chi tiết
        - Expandable: Có thể mở rộng ra parent khi cần
        """
        query_embedding = self.embedding_model.embed_query(query)
        
        # Retrieve child chunks
        results = self.vectorstore.similarity_search_with_ids(
            query_embedding,
            k=self.child_k
        )
        
        retrieval_results = []
        
        for doc_id, text, score, metadata in results:
            chunk_info = self.chunk_map.get(doc_id, {})
            parent_id = chunk_info.get("parent_id")
            
            # Lấy parent content nếu có
            parent_content = None
            if parent_id:
                parent_info = self.chunk_map.get(parent_id)
                if parent_info:
                    parent_content = parent_info["content"]
            
            result = RetrievalResult(
                content=text,
                score=score,
                metadata={
                    **metadata,
                    "retrieval_strategy": "child_first"
                },
                chunk_id=doc_id,
                parent_content=parent_content,
                level=chunk_info.get("level")
            )
            retrieval_results.append(result)
            
            if len(retrieval_results) >= k:
                break
        
        return retrieval_results[:k]
    
    def _retrieve_parent_first(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Parent-First: Retrieve parent chunks trước,
        kèm theo tất cả children chunks
        
        Strategy này tốt cho:
        - Broad context: Lấy ngữ cảnh rộng trước
        - Complete info: Có đầy đủ thông tin từ parent và children
        """
        query_embedding = self.embedding_model.embed_query(query)
        
        # Retrieve parent chunks (filter by level if possible)
        results = self.vectorstore.similarity_search_with_ids(
            query_embedding,
            k=self.parent_k
        )
        
        retrieval_results = []
        
        for doc_id, text, score, metadata in results:
            chunk_info = self.chunk_map.get(doc_id, {})
            children_ids = chunk_info.get("children_ids", [])
            
            # Lấy content của tất cả children
            children_contents = []
            for child_id in children_ids:
                child_info = self.chunk_map.get(child_id)
                if child_info:
                    children_contents.append(child_info["content"])
            
            result = RetrievalResult(
                content=text,
                score=score,
                metadata={
                    **metadata,
                    "retrieval_strategy": "parent_first",
                    "children_count": len(children_contents)
                },
                chunk_id=doc_id,
                children_contents=children_contents if children_contents else None,
                level=chunk_info.get("level")
            )
            retrieval_results.append(result)
            
            if len(retrieval_results) >= k:
                break
        
        return retrieval_results[:k]
    
    def _retrieve_multi_level(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Multi-Level: Retrieve ở nhiều levels khác nhau đồng thời
        và merge results
        
        Strategy này tốt cho:
        - Comprehensive: Lấy từ nhiều góc độ
        - Balanced: Cân bằng giữa detail và context
        """
        query_embedding = self.embedding_model.embed_query(query)
        
        # Retrieve nhiều hơn để có thể filter và merge
        results = self.vectorstore.similarity_search_with_ids(
            query_embedding,
            k=k * 3
        )
        
        # Group by level
        level_groups = {}
        for doc_id, text, score, metadata in results:
            chunk_info = self.chunk_map.get(doc_id, {})
            level = chunk_info.get("level", "unknown")
            
            if level not in level_groups:
                level_groups[level] = []
            
            level_groups[level].append((doc_id, text, score, metadata, chunk_info))
        
        # Merge results from different levels
        retrieval_results = []
        
        # Ưu tiên PARAGRAPH và SENTENCE levels
        priority_levels = ["PARAGRAPH", "SENTENCE", "SECTION", "DOCUMENT"]
        
        for level in priority_levels:
            if level not in level_groups:
                continue
            
            for doc_id, text, score, metadata, chunk_info in level_groups[level]:
                parent_id = chunk_info.get("parent_id")
                parent_content = None
                
                if parent_id:
                    parent_info = self.chunk_map.get(parent_id)
                    if parent_info:
                        parent_content = parent_info["content"]
                
                result = RetrievalResult(
                    content=text,
                    score=score,
                    metadata={
                        **metadata,
                        "retrieval_strategy": "multi_level"
                    },
                    chunk_id=doc_id,
                    parent_content=parent_content,
                    level=level
                )
                retrieval_results.append(result)
                
                if len(retrieval_results) >= k:
                    break
            
            if len(retrieval_results) >= k:
                break
        
        return retrieval_results[:k]


class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever kết hợp nhiều retrieval methods
    Ví dụ: Dense retrieval (vector) + Sparse retrieval (BM25) + Reranking
    """
    
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
        reranker=None
    ):
        """
        Args:
            retrievers: Danh sách các retrievers
            weights: Weights cho mỗi retriever (optional)
            reranker: Reranker model (optional)
        """
        self.retrievers = retrievers
        
        if weights is None:
            # Equal weights
            weights = [1.0 / len(retrievers)] * len(retrievers)
        
        if len(weights) != len(retrievers):
            raise ValueError("Số weights phải bằng số retrievers")
        
        self.weights = weights
        self.reranker = reranker
    
    def retrieve(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve với hybrid approach
        
        Args:
            query: Query string
            k: Số kết quả cuối cùng
            
        Returns:
            List[RetrievalResult]: Merged và reranked results
        """
        # Retrieve từ tất cả retrievers
        all_results = []
        
        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.retrieve(query, k=k * 2)
            
            # Adjust scores với weight
            for result in results:
                result.score *= weight
                all_results.append(result)
        
        # Merge results (remove duplicates based on chunk_id)
        merged_results = self._merge_results(all_results)
        
        # Rerank nếu có reranker
        if self.reranker:
            merged_results = self._rerank(query, merged_results)
        
        # Sort by score và return top k
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results[:k]
    
    def _merge_results(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Merge results, combine scores cho duplicate chunks
        """
        chunk_scores = {}
        chunk_map = {}
        
        for result in results:
            chunk_id = result.chunk_id or result.content
            
            if chunk_id in chunk_scores:
                # Combine scores (có thể dùng sum, max, hoặc average)
                chunk_scores[chunk_id] += result.score
            else:
                chunk_scores[chunk_id] = result.score
                chunk_map[chunk_id] = result
        
        # Update scores
        merged = []
        for chunk_id, combined_score in chunk_scores.items():
            result = chunk_map[chunk_id]
            result.score = combined_score
            merged.append(result)
        
        return merged
    
    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Rerank results sử dụng reranker model
        """
        if not results or not self.reranker:
            return results
        
        # Prepare pairs for reranker
        pairs = [(query, result.content) for result in results]
        
        # Get rerank scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update scores
        for result, new_score in zip(results, rerank_scores):
            result.score = new_score
            result.metadata["original_score"] = result.score
            result.metadata["reranked"] = True
        
        return results


def build_chunk_map_from_hierarchical_chunks(
    hierarchical_chunks
) -> Dict[str, Dict[str, Any]]:
    """
    Utility function để build chunk_map từ HierarchicalChunk objects
    
    Args:
        hierarchical_chunks: List of HierarchicalChunk objects
        
    Returns:
        Dict[str, Dict]: Mapping chunk_id -> chunk info
    """
    chunk_map = {}
    
    for chunk in hierarchical_chunks:
        chunk_map[chunk.chunk_id] = {
            "content": chunk.content,
            "level": chunk.level.name,
            "parent_id": chunk.parent_id,
            "children_ids": chunk.children_ids,
            "metadata": chunk.metadata,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char
        }
    
    return chunk_map
