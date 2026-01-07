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
    def search(
        self,
        query: Union[str, List[float]],
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[Tuple[str, float, Dict]]:
        """
        Phương thức search tổng quát - có thể nhận query text hoặc embedding
        
        Args:
            query: Query string hoặc query embedding vector
            k: Số lượng kết quả trả về
            filter: Filter metadata (optional)
            **kwargs: Các tham số bổ sung tùy implementation
            
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
    
    @abstractmethod
    def clear(self) -> None:
        """
        Xóa tất cả dữ liệu trong vector store
        """
        pass

    