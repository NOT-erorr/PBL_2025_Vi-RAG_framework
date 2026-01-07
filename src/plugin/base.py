from abc import ABC, abstractmethod
from typing import Any, Dict

class BasePlugin(ABC):
    """
    Lớp trừu tượng cho mọi Plugin/Tool trong hệ thống RAG.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tên định danh của tool (vd: 'google_search')"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Mô tả chức năng để LLM hiểu khi nào cần dùng.
        Vd: 'Dùng tool này để tìm kiếm thông tin thời gian thực.'
        """
        pass

    @abstractmethod
    def run(self, input_query: str, **kwargs) -> Any:
        """Thực thi logic của tool"""
        pass
        
    def to_tool_schema(self) -> Dict:
        """Chuyển đổi plugin thành format mà LLM (Function Calling) hiểu được"""
        return {
            "name": self.name,
            "description": self.description,
            # Có thể mở rộng thêm parameters schema ở đây nếu cần
        }