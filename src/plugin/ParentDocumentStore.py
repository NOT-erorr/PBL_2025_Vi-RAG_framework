"""
Parent Document Store Plugin

Implementation of Parent Document Retrieval Strategy for RAG systems.
Provides classes for managing parent-child chunk relationships.

Classes:
    - ParentChildChunk: Dataclass for parent-child chunk with metadata
    - ParentDocumentStore: Storage system for parent chunks and mappings
    - ParentChildTextSplitter: Splitter for creating parent-child chunks

Author: PBL-2025 Team
Date: 2025-12-28
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import uuid
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import sentence tokenizer
try:
    from underthesea import sent_tokenize
except ImportError:
    import re
    def sent_tokenize(text: str) -> List[str]:
        """Fallback sentence tokenizer using regex"""
        pattern = re.compile(r'[^.!?]+[.!?]+')
        sentences = pattern.findall(text)
        return [s.strip() for s in sentences] if sentences else [text]


# Import Document class from core
try:
    from core.splitter import Document
except ImportError:
    # Fallback Document class nếu không import được
    class Document:
        """Lớp đại diện cho một tài liệu"""
        def __init__(self, content: str, metadata: dict = None):
            self.content = content
            self.metadata = metadata or {}


@dataclass
class ParentChildChunk:
    """
    Dataclass đại diện cho một chunk với parent-child relationship
    Sử dụng cho Parent Document Retrieval strategy
    """
    child_id: str                    # ID của child chunk (được index và search)
    child_content: str               # Nội dung của child chunk (nhỏ, dùng cho retrieval)
    parent_id: str                   # ID của parent chunk
    parent_content: str              # Nội dung của parent chunk (lớn hơn, cung cấp context)
    metadata: Dict = field(default_factory=dict)
    child_start_char: int = 0        # Vị trí bắt đầu của child trong parent
    child_end_char: int = 0          # Vị trí kết thúc của child trong parent


class ParentDocumentStore:
    """
    Storage cho Parent Document Retrieval Strategy
    
    Chiến lược này chia documents thành:
    - Child chunks: Nhỏ hơn, được index để tìm kiếm chính xác
    - Parent chunks: Lớn hơn, chứa context đầy đủ, được trả về khi retrieve
    
    Flow:
    1. Chia document thành parent chunks (lớn)
    2. Mỗi parent chunk được chia tiếp thành child chunks (nhỏ)
    3. Child chunks được index trong vector store để search
    4. Khi retrieve, trả về parent chunk chứa child chunk được match
    
    Ưu điểm:
    - Tìm kiếm chính xác với child chunks nhỏ
    - Context đầy đủ hơn khi trả về parent chunks lớn
    - Giảm thiểu mất mát thông tin do chunking
    
    Example:
        >>> store = ParentDocumentStore()
        >>> parent_child_splitter = ParentChildTextSplitter(
        ...     parent_chunk_size=2000,
        ...     child_chunk_size=400
        ... )
        >>> chunks = parent_child_splitter.split_text(text)
        >>> store.add_chunks(chunks)
        >>> store.save("parent_store.json")
    """
    
    def __init__(self):
        """Khởi tạo Parent Document Store"""
        self.parent_chunks: Dict[str, str] = {}          # parent_id -> parent_content
        self.child_to_parent: Dict[str, str] = {}        # child_id -> parent_id
        self.child_metadata: Dict[str, Dict] = {}        # child_id -> metadata
        self.parent_metadata: Dict[str, Dict] = {}       # parent_id -> metadata
        
        logger.info("Initialized ParentDocumentStore")
    
    def add_chunks(
        self, 
        chunks: List[ParentChildChunk],
        overwrite: bool = False
    ) -> None:
        """
        Thêm parent-child chunks vào store
        
        Args:
            chunks: Danh sách ParentChildChunk objects
            overwrite: Có ghi đè nếu ID đã tồn tại (mặc định: False)
        
        Raises:
            ValueError: Nếu chunk ID đã tồn tại và overwrite=False
        
        Example:
            >>> chunks = splitter.split_text("Long document...")
            >>> store.add_chunks(chunks)
            >>> print(f"Added {len(chunks)} chunks")
        """
        added_parents = set()
        added_children = 0
        
        for chunk in chunks:
            # Check duplicates
            if not overwrite:
                if chunk.child_id in self.child_to_parent:
                    raise ValueError(f"Child ID already exists: {chunk.child_id}")
            
            # Store parent content (chỉ lưu một lần cho mỗi parent)
            if chunk.parent_id not in self.parent_chunks:
                self.parent_chunks[chunk.parent_id] = chunk.parent_content
                self.parent_metadata[chunk.parent_id] = chunk.metadata.copy()
                added_parents.add(chunk.parent_id)
            
            # Map child to parent
            self.child_to_parent[chunk.child_id] = chunk.parent_id
            
            # Store child metadata
            child_meta = chunk.metadata.copy()
            child_meta.update({
                'child_start': chunk.child_start_char,
                'child_end': chunk.child_end_char,
                'child_length': len(chunk.child_content),
                'parent_length': len(chunk.parent_content)
            })
            self.child_metadata[chunk.child_id] = child_meta
            
            added_children += 1
        
        logger.info(
            f"Added {added_children} child chunks from {len(added_parents)} parent chunks"
        )
    
    def get_parent_by_child_id(self, child_id: str) -> Optional[str]:
        """
        Lấy parent content từ child ID
        
        Args:
            child_id: ID của child chunk
            
        Returns:
            Optional[str]: Parent content hoặc None nếu không tìm thấy
            
        Example:
            >>> parent = store.get_parent_by_child_id("child_123")
            >>> if parent:
            ...     print(f"Parent content: {parent[:100]}...")
        """
        parent_id = self.child_to_parent.get(child_id)
        if parent_id:
            return self.parent_chunks.get(parent_id)
        return None
    
    def get_parent_with_metadata(
        self, 
        child_id: str
    ) -> Optional[Tuple[str, Dict]]:
        """
        Lấy parent content và metadata từ child ID
        
        Args:
            child_id: ID của child chunk
            
        Returns:
            Optional[Tuple[str, Dict]]: (parent_content, metadata) hoặc None
            
        Example:
            >>> result = store.get_parent_with_metadata("child_123")
            >>> if result:
            ...     content, metadata = result
            ...     print(f"Parent from: {metadata['source']}")
        """
        parent_id = self.child_to_parent.get(child_id)
        if parent_id:
            parent_content = self.parent_chunks.get(parent_id)
            parent_meta = self.parent_metadata.get(parent_id, {})
            child_meta = self.child_metadata.get(child_id, {})
            
            # Merge metadata
            combined_meta = parent_meta.copy()
            combined_meta.update({
                'child_id': child_id,
                'parent_id': parent_id,
                **child_meta
            })
            
            return (parent_content, combined_meta)
        return None
    
    def get_parents_by_child_ids(
        self, 
        child_ids: List[str],
        deduplicate: bool = True
    ) -> List[Tuple[str, Dict]]:
        """
        Lấy parent contents cho nhiều child IDs
        
        Args:
            child_ids: Danh sách child IDs
            deduplicate: Loại bỏ parent trùng lặp (mặc định: True)
            
        Returns:
            List[Tuple[str, Dict]]: Danh sách (parent_content, metadata)
            
        Example:
            >>> child_ids = ["child_1", "child_2", "child_3"]
            >>> parents = store.get_parents_by_child_ids(child_ids)
            >>> print(f"Retrieved {len(parents)} unique parent chunks")
        """
        results = []
        seen_parent_ids = set()
        
        for child_id in child_ids:
            result = self.get_parent_with_metadata(child_id)
            if result:
                parent_content, metadata = result
                parent_id = metadata.get('parent_id')
                
                # Deduplicate nếu cần
                if deduplicate:
                    if parent_id not in seen_parent_ids:
                        results.append((parent_content, metadata))
                        seen_parent_ids.add(parent_id)
                else:
                    results.append((parent_content, metadata))
        
        return results
    
    def get_child_metadata(self, child_id: str) -> Optional[Dict]:
        """
        Lấy metadata của child chunk
        
        Args:
            child_id: ID của child chunk
            
        Returns:
            Optional[Dict]: Child metadata hoặc None
        """
        return self.child_metadata.get(child_id)
    
    def get_parent_metadata(self, parent_id: str) -> Optional[Dict]:
        """
        Lấy metadata của parent chunk
        
        Args:
            parent_id: ID của parent chunk
            
        Returns:
            Optional[Dict]: Parent metadata hoặc None
        """
        return self.parent_metadata.get(parent_id)
    
    def get_children_by_parent_id(self, parent_id: str) -> List[str]:
        """
        Lấy tất cả child IDs của một parent
        
        Args:
            parent_id: ID của parent chunk
            
        Returns:
            List[str]: Danh sách child IDs
            
        Example:
            >>> children = store.get_children_by_parent_id("parent_123")
            >>> print(f"Parent has {len(children)} children")
        """
        return [
            child_id for child_id, pid in self.child_to_parent.items()
            if pid == parent_id
        ]
    
    def count_parents(self) -> int:
        """
        Đếm số lượng parent chunks
        
        Returns:
            int: Số lượng parent chunks
        """
        return len(self.parent_chunks)
    
    def count_children(self) -> int:
        """
        Đếm số lượng child chunks
        
        Returns:
            int: Số lượng child chunks
        """
        return len(self.child_to_parent)
    
    def get_statistics(self) -> Dict:
        """
        Lấy thống kê về parent document store
        
        Returns:
            Dict: Statistics dictionary
            
        Example:
            >>> stats = store.get_statistics()
            >>> print(f"Total parents: {stats['total_parents']}")
            >>> print(f"Avg children per parent: {stats['avg_children_per_parent']:.2f}")
        """
        total_parents = self.count_parents()
        total_children = self.count_children()
        
        # Tính children per parent
        children_per_parent = {}
        for child_id, parent_id in self.child_to_parent.items():
            children_per_parent[parent_id] = children_per_parent.get(parent_id, 0) + 1
        
        avg_children = (
            sum(children_per_parent.values()) / len(children_per_parent)
            if children_per_parent else 0
        )
        
        # Tính độ dài trung bình
        avg_parent_length = (
            sum(len(content) for content in self.parent_chunks.values()) / total_parents
            if total_parents > 0 else 0
        )
        
        return {
            'total_parents': total_parents,
            'total_children': total_children,
            'avg_children_per_parent': avg_children,
            'avg_parent_length': avg_parent_length,
            'unique_parents': len(set(self.child_to_parent.values()))
        }
    
    def save(self, path: str) -> None:
        """
        Lưu parent document store vào file JSON
        
        Args:
            path: Đường dẫn file để lưu
            
        Raises:
            IOError: Nếu không thể ghi file
            
        Example:
            >>> store.save("data/parent_store.json")
            >>> print("Parent store saved successfully")
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'parent_chunks': self.parent_chunks,
            'child_to_parent': self.child_to_parent,
            'child_metadata': self.child_metadata,
            'parent_metadata': self.parent_metadata
        }
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved ParentDocumentStore to {path}")
            logger.info(
                f"  - {len(self.parent_chunks)} parents, "
                f"{len(self.child_to_parent)} children"
            )
        except Exception as e:
            logger.error(f"Failed to save ParentDocumentStore: {e}")
            raise
    
    def load(self, path: str) -> None:
        """
        Load parent document store từ file JSON
        
        Args:
            path: Đường dẫn file để load
            
        Raises:
            FileNotFoundError: Nếu file không tồn tại
            IOError: Nếu không thể đọc file
            
        Example:
            >>> store = ParentDocumentStore()
            >>> store.load("data/parent_store.json")
            >>> print(f"Loaded {store.count_parents()} parents")
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.parent_chunks = data['parent_chunks']
            self.child_to_parent = data['child_to_parent']
            self.child_metadata = data['child_metadata']
            self.parent_metadata = data['parent_metadata']
            
            logger.info(f"Loaded ParentDocumentStore from {path}")
            logger.info(
                f"  - {len(self.parent_chunks)} parents, "
                f"{len(self.child_to_parent)} children"
            )
        except Exception as e:
            logger.error(f"Failed to load ParentDocumentStore: {e}")
            raise
    
    def clear(self) -> None:
        """
        Xóa tất cả dữ liệu trong store
        
        Example:
            >>> store.clear()
            >>> print("Store cleared")
        """
        self.parent_chunks.clear()
        self.child_to_parent.clear()
        self.child_metadata.clear()
        self.parent_metadata.clear()
        
        logger.info("Cleared ParentDocumentStore")


class ParentChildTextSplitter:
    """
    Text Splitter cho Parent Document Retrieval Strategy
    
    Chia document thành parent chunks (lớn) và child chunks (nhỏ) với relationship
    
    Strategy:
    1. Chia document thành parent chunks với kích thước lớn
    2. Mỗi parent chunk được chia tiếp thành child chunks nhỏ hơn
    3. Mỗi child chunk giữ reference đến parent chunk của nó
    
    Args:
        parent_chunk_size: Kích thước của parent chunks (lớn, cho context)
        child_chunk_size: Kích thước của child chunks (nhỏ, cho retrieval)
        parent_overlap: Overlap giữa các parent chunks
        child_overlap: Overlap giữa các child chunks trong cùng parent
        preserve_sentences: Giữ nguyên câu hoàn chỉnh khi split
        
    Example:
        >>> splitter = ParentChildTextSplitter(
        ...     parent_chunk_size=2000,
        ...     child_chunk_size=400,
        ...     parent_overlap=200,
        ...     child_overlap=50
        ... )
        >>> text = "Your long document..."
        >>> chunks = splitter.split_text(text, source="document.pdf")
        >>> print(f"Created {len(chunks)} child chunks")
        >>> 
        >>> # Use with store
        >>> store = ParentDocumentStore()
        >>> store.add_chunks(chunks)
    """
    
    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 400,
        parent_overlap: int = 200,
        child_overlap: int = 50,
        preserve_sentences: bool = True
    ):
        """Khởi tạo Parent-Child Text Splitter"""
        if child_chunk_size >= parent_chunk_size:
            raise ValueError(
                "child_chunk_size phải nhỏ hơn parent_chunk_size"
            )
        
        if parent_overlap >= parent_chunk_size:
            raise ValueError(
                "parent_overlap phải nhỏ hơn parent_chunk_size"
            )
        
        if child_overlap >= child_chunk_size:
            raise ValueError(
                "child_overlap phải nhỏ hơn child_chunk_size"
            )
        
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        self.preserve_sentences = preserve_sentences
        
        logger.info(
            f"Initialized ParentChildTextSplitter: "
            f"parent={parent_chunk_size}, child={child_chunk_size}"
        )
    
    def split_text(
        self, 
        text: str,
        source: str = "unknown",
        base_metadata: Optional[Dict] = None
    ) -> List[ParentChildChunk]:
        """
        Split text thành parent-child chunks
        
        Args:
            text: Text cần split
            source: Tên nguồn document (cho metadata)
            base_metadata: Metadata cơ bản để thêm vào các chunks
            
        Returns:
            List[ParentChildChunk]: Danh sách parent-child chunks
            
        Example:
            >>> text = load_document("long_doc.txt")
            >>> chunks = splitter.split_text(text, source="long_doc.txt")
            >>> print(f"Split into {len(chunks)} child chunks")
        """
        if not text or len(text.strip()) == 0:
            return []
        
        base_metadata = base_metadata or {}
        all_chunks = []
        
        # Step 1: Split thành parent chunks
        parent_chunks = self._split_into_parents(text)
        
        # Step 2: Với mỗi parent, split thành child chunks
        for parent_idx, (parent_text, parent_start, parent_end) in enumerate(parent_chunks):
            parent_id = f"parent_{uuid.uuid4().hex[:8]}_{parent_idx}"
            
            # Split parent thành children
            child_chunks = self._split_into_children(parent_text)
            
            # Create ParentChildChunk objects
            for child_idx, (child_text, child_start, child_end) in enumerate(child_chunks):
                child_id = f"child_{uuid.uuid4().hex[:8]}_{parent_idx}_{child_idx}"
                
                metadata = base_metadata.copy()
                metadata.update({
                    'source': source,
                    'parent_index': parent_idx,
                    'child_index': child_idx,
                    'total_children_in_parent': len(child_chunks),
                    'parent_global_start': parent_start,
                    'parent_global_end': parent_end
                })
                
                chunk = ParentChildChunk(
                    child_id=child_id,
                    child_content=child_text,
                    parent_id=parent_id,
                    parent_content=parent_text,
                    metadata=metadata,
                    child_start_char=child_start,
                    child_end_char=child_end
                )
                
                all_chunks.append(chunk)
        
        logger.info(
            f"Split text into {len(parent_chunks)} parents "
            f"and {len(all_chunks)} children"
        )
        
        return all_chunks
    
    def _split_into_parents(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text thành parent chunks
        
        Returns:
            List[Tuple[str, int, int]]: (chunk_text, start_pos, end_pos)
        """
        if not self.preserve_sentences:
            # Simple character-based splitting
            chunks = []
            pos = 0
            while pos < len(text):
                end = min(pos + self.parent_chunk_size, len(text))
                chunks.append((text[pos:end], pos, end))
                pos = end - self.parent_overlap
            return chunks
        
        # Sentence-based splitting
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        current_start = 0
        current_pos = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_length = len(sent)
            
            # Nếu thêm câu này vượt quá size
            if current_length + sent_length + 1 > self.parent_chunk_size and current_chunk:
                # Lưu chunk hiện tại
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, current_start, current_pos))
                
                # Tính overlap (lấy lại một số câu cuối)
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.parent_overlap:
                        overlap_sents.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break
                
                current_chunk = overlap_sents + [sent]
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk)
                current_start = current_pos - overlap_len
            else:
                current_chunk.append(sent)
                current_length += sent_length + 1
            
            current_pos += sent_length + 1
        
        # Thêm chunk cuối
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, current_start, len(text)))
        
        return chunks if chunks else [(text, 0, len(text))]
    
    def _split_into_children(self, parent_text: str) -> List[Tuple[str, int, int]]:
        """
        Split parent chunk thành child chunks
        
        Returns:
            List[Tuple[str, int, int]]: (chunk_text, start_pos, end_pos)
        """
        if not self.preserve_sentences:
            # Simple character-based splitting
            chunks = []
            pos = 0
            while pos < len(parent_text):
                end = min(pos + self.child_chunk_size, len(parent_text))
                chunks.append((parent_text[pos:end], pos, end))
                pos = end - self.child_overlap
            return chunks
        
        # Sentence-based splitting
        sentences = sent_tokenize(parent_text)
        chunks = []
        current_chunk = []
        current_length = 0
        current_start = 0
        current_pos = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_length = len(sent)
            
            if current_length + sent_length + 1 > self.child_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, current_start, current_pos))
                
                # Overlap
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.child_overlap:
                        overlap_sents.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break
                
                current_chunk = overlap_sents + [sent]
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk)
                current_start = current_pos - overlap_len
            else:
                current_chunk.append(sent)
                current_length += sent_length + 1
            
            current_pos += sent_length + 1
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, current_start, len(parent_text)))
        
        return chunks if chunks else [(parent_text, 0, len(parent_text))]
    
    def split_documents(
        self, 
        documents: List[Document]
    ) -> List[ParentChildChunk]:
        """
        Split danh sách documents thành parent-child chunks
        
        Args:
            documents: Danh sách Document objects
            
        Returns:
            List[ParentChildChunk]: Danh sách parent-child chunks
            
        Example:
            >>> from core.loader import TXTLoader
            >>> loader = TXTLoader("document.txt")
            >>> docs = loader.load()
            >>> chunks = splitter.split_documents(docs)
        """
        all_chunks = []
        
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            chunks = self.split_text(
                doc.content, 
                source=source,
                base_metadata=doc.metadata
            )
            all_chunks.extend(chunks)
        
        return all_chunks
