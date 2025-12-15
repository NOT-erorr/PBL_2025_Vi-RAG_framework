from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
try:
    from underthesea import sent_tokenize
except ImportError:
    # Fallback nếu underthesea chưa được cài đặt
    import re
    def sent_tokenize(text: str) -> List[str]:
        """Fallback sentence tokenizer using regex"""
        pattern = re.compile(r'[^.!?]+[.!?]+')
        sentences = pattern.findall(text)
        return [s.strip() for s in sentences] if sentences else [text]


class ChunkLevel(Enum):
    """Enum định nghĩa các mức độ chunk trong hierarchy"""
    DOCUMENT = 0      # Toàn bộ document
    SECTION = 1       # Section/Chapter lớn
    PARAGRAPH = 2     # Đoạn văn
    SENTENCE = 3      # Câu
    SUBSECTION = 4    # Phần nhỏ hơn nếu cần


@dataclass
class HierarchicalChunk:
    """
    Dataclass đại diện cho một chunk trong cấu trúc phân cấp
    """
    content: str
    level: ChunkLevel
    chunk_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0


class Document:
    """Lớp đại diện cho một tài liệu"""
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}



class TextSplitter(ABC):
    """Abstract class cho text splitter"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: Kích thước tối đa của mỗi chunk
            chunk_overlap: Số ký tự overlap giữa các chunks
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap phải nhỏ hơn chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split text thành các chunks
        
        Args:
            text: Text cần split
            
        Returns:
            List[str]: Danh sách các text chunks
        """
        pass
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split danh sách documents thành các chunks nhỏ hơn
        
        Args:
            documents: Danh sách documents cần split
            
        Returns:
            List[Document]: Danh sách documents đã được split
        """
        split_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata['chunk_index'] = i
                metadata['total_chunks'] = len(chunks)
                split_docs.append(Document(content=chunk, metadata=metadata))
        
        return split_docs


class VietnameseTextSplitter(TextSplitter):
    """Text splitter tối ưu cho tiếng Việt sử dụng underthesea"""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        keep_sentence_integrity: bool = True
    ):
        """
        Args:
            chunk_size: Kích thước tối đa của mỗi chunk
            chunk_overlap: Số ký tự overlap giữa các chunks
            keep_sentence_integrity: Giữ nguyên câu hoàn chỉnh trong chunk (mặc định: True)
        """
        super().__init__(chunk_size, chunk_overlap)
        self.keep_sentence_integrity = keep_sentence_integrity
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text tiếng Việt thành các chunks, giữ nguyên ngữ cảnh và câu hoàn chỉnh
        
        Args:
            text: Text tiếng Việt cần split
            
        Returns:
            List[str]: Danh sách các text chunks với ngữ cảnh được bảo toàn
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Sử dụng underthesea để tách câu chính xác
        sentences = sent_tokenize(text)
        
        if not sentences:
            return [text]
        
        return self._merge_sentences_into_chunks(sentences)
    
    def _merge_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """
        Merge các câu thành chunks, đảm bảo giữ nguyên câu hoàn chỉnh và ngữ cảnh
        
        Args:
            sentences: Danh sách các câu đã được tokenize
            
        Returns:
            List[str]: Danh sách các chunks với câu hoàn chỉnh
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            sentence_length = len(sentence)
            
            # Nếu câu đơn lẻ vượt quá chunk_size
            if sentence_length > self.chunk_size:
                # Lưu chunk hiện tại
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split câu dài thành các phần nhỏ hơn (chỉ khi cần thiết)
                for j in range(0, sentence_length, self.chunk_size):
                    sub_chunk = sentence[j:j + self.chunk_size]
                    if sub_chunk.strip():
                        chunks.append(sub_chunk)
                continue
            
            # Nếu thêm câu vào chunk hiện tại vẫn còn trong giới hạn
            if current_length + sentence_length + 1 <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                # Lưu chunk hiện tại và bắt đầu chunk mới
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Thêm chunk cuối cùng
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Thêm overlap với câu hoàn chỉnh
        return self._add_sentence_overlap(chunks, sentences)
    
    def _add_sentence_overlap(self, chunks: List[str], sentences: List[str]) -> List[str]:
        """
        Thêm overlap giữa các chunks bằng cách thêm câu hoàn chỉnh từ chunk trước
        Giữ nguyên ngữ cảnh bằng cách overlap theo câu thay vì ký tự
        
        Args:
            chunks: Danh sách chunks ban đầu
            sentences: Danh sách câu gốc
            
        Returns:
            List[str]: Danh sách chunks có overlap với câu hoàn chỉnh
        """
        if not chunks or self.chunk_overlap == 0 or not self.keep_sentence_integrity:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # Tách câu từ chunk trước để lấy overlap
            prev_sentences = sent_tokenize(prev_chunk)
            
            # Tìm các câu từ cuối chunk trước để thêm vào overlap
            overlap_sentences = []
            overlap_length = 0
            
            for sent in reversed(prev_sentences):
                sent_length = len(sent)
                if overlap_length + sent_length <= self.chunk_overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_length += sent_length + 1
                else:
                    break
            
            # Kết hợp overlap với chunk hiện tại
            if overlap_sentences:
                new_chunk = " ".join(overlap_sentences) + " " + current_chunk
                overlapped_chunks.append(new_chunk)
            else:
                overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks


class CharacterTextSplitter(TextSplitter):
    """Simple text splitter theo số ký tự"""
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text theo số ký tự cố định
        
        Args:
            text: Text cần split
            
        Returns:
            List[str]: Danh sách các chunks
        """
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            # Di chuyển start position với overlap
            start = end - self.chunk_overlap if end < text_length else text_length
        
        return chunks


class SentenceTextSplitter(TextSplitter):
    """Text splitter theo câu sử dụng underthesea (sentence-based)"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        super().__init__(chunk_size, chunk_overlap)
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text theo câu sử dụng underthesea sent_tokenize
        
        Args:
            text: Text cần split
            
        Returns:
            List[str]: Danh sách các chunks với câu hoàn chỉnh
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Tách thành các câu bằng underthesea
        sentences = sent_tokenize(text)
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_length = len(sentence)
            
            if current_length + sentence_length + 1 <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Thêm chunk cuối
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Thêm overlap với câu hoàn chỉnh
        return self._add_overlap_sentences(chunks)
    
    def _add_overlap_sentences(self, chunks: List[str]) -> List[str]:
        """
        Thêm overlap giữa các chunks với câu hoàn chỉnh
        
        Args:
            chunks: Danh sách chunks
            
        Returns:
            List[str]: Chunks có overlap
        """
        if not chunks or self.chunk_overlap == 0:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_sentences = sent_tokenize(chunks[i - 1])
            
            # Lấy các câu cuối từ chunk trước
            overlap_sents = []
            overlap_len = 0
            
            for sent in reversed(prev_sentences):
                if overlap_len + len(sent) <= self.chunk_overlap:
                    overlap_sents.insert(0, sent)
                    overlap_len += len(sent) + 1
                else:
                    break
            
            if overlap_sents:
                new_chunk = " ".join(overlap_sents) + " " + chunks[i]
                overlapped.append(new_chunk)
            else:
                overlapped.append(chunks[i])
        
        return overlapped


class HierarchicalTextSplitter:
    """
    Hierarchical Text Splitter cho Vietnamese documents
    Chia text thành nhiều levels với parent-child relationships
    """
    
    def __init__(
        self,
        section_size: int = 5000,
        paragraph_size: int = 1000,
        sentence_size: int = 200,
        overlap: int = 100
    ):
        """
        Args:
            section_size: Kích thước tối đa của section (level 1)
            paragraph_size: Kích thước tối đa của paragraph (level 2)
            sentence_size: Kích thước tối đa của sentence group (level 3)
            overlap: Overlap giữa các chunks cùng level
        """
        self.section_size = section_size
        self.paragraph_size = paragraph_size
        self.sentence_size = sentence_size
        self.overlap = overlap
        self._chunk_counter = 0
    
    def _generate_chunk_id(self, level: ChunkLevel, parent_id: Optional[str] = None) -> str:
        """Generate unique chunk ID"""
        self._chunk_counter += 1
        if parent_id:
            return f"{parent_id}.{level.name}_{self._chunk_counter}"
        return f"{level.name}_{self._chunk_counter}"
    
    def split_text(self, text: str, source_name: str = "document") -> List[HierarchicalChunk]:
        """
        Split text thành hierarchical chunks
        
        Args:
            text: Text cần split
            source_name: Tên nguồn document
            
        Returns:
            List[HierarchicalChunk]: Danh sách các chunks với cấu trúc phân cấp
        """
        if not text or len(text.strip()) == 0:
            return []
        
        self._chunk_counter = 0
        all_chunks = []
        
        # Level 0: Document level
        doc_id = self._generate_chunk_id(ChunkLevel.DOCUMENT)
        doc_chunk = HierarchicalChunk(
            content=text,
            level=ChunkLevel.DOCUMENT,
            chunk_id=doc_id,
            parent_id=None,
            metadata={"source": source_name, "total_length": len(text)},
            start_char=0,
            end_char=len(text)
        )
        all_chunks.append(doc_chunk)
        
        # Level 1: Section level (split by double newlines or large paragraphs)
        sections = self._split_into_sections(text)
        section_chunks = []
        
        for section_text, start_pos in sections:
            section_id = self._generate_chunk_id(ChunkLevel.SECTION, doc_id)
            section_chunk = HierarchicalChunk(
                content=section_text,
                level=ChunkLevel.SECTION,
                chunk_id=section_id,
                parent_id=doc_id,
                metadata={"source": source_name},
                start_char=start_pos,
                end_char=start_pos + len(section_text)
            )
            section_chunks.append(section_chunk)
            doc_chunk.children_ids.append(section_id)
            all_chunks.append(section_chunk)
            
            # Level 2: Paragraph level
            paragraphs = self._split_into_paragraphs(section_text, start_pos)
            
            for para_text, para_start in paragraphs:
                para_id = self._generate_chunk_id(ChunkLevel.PARAGRAPH, section_id)
                para_chunk = HierarchicalChunk(
                    content=para_text,
                    level=ChunkLevel.PARAGRAPH,
                    chunk_id=para_id,
                    parent_id=section_id,
                    metadata={"source": source_name},
                    start_char=para_start,
                    end_char=para_start + len(para_text)
                )
                section_chunk.children_ids.append(para_id)
                all_chunks.append(para_chunk)
                
                # Level 3: Sentence level
                sentences = self._split_into_sentences(para_text, para_start)
                
                for sent_text, sent_start in sentences:
                    sent_id = self._generate_chunk_id(ChunkLevel.SENTENCE, para_id)
                    sent_chunk = HierarchicalChunk(
                        content=sent_text,
                        level=ChunkLevel.SENTENCE,
                        chunk_id=sent_id,
                        parent_id=para_id,
                        metadata={"source": source_name},
                        start_char=sent_start,
                        end_char=sent_start + len(sent_text)
                    )
                    para_chunk.children_ids.append(sent_id)
                    all_chunks.append(sent_chunk)
        
        return all_chunks
    
    def _split_into_sections(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text thành sections (các phần lớn)
        
        Returns:
            List[Tuple[str, int]]: List of (section_text, start_position)
        """
        # Split theo \n\n (paragraph breaks) và group thành sections
        paragraphs = text.split('\n\n')
        sections = []
        current_section = []
        current_length = 0
        current_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                current_pos += 2  # \n\n
                continue
            
            para_length = len(para)
            
            if current_length + para_length + 2 <= self.section_size:
                current_section.append(para)
                current_length += para_length + 2
            else:
                if current_section:
                    section_text = '\n\n'.join(current_section)
                    section_start = current_pos - current_length
                    sections.append((section_text, section_start))
                
                current_section = [para]
                current_length = para_length
            
            current_pos += para_length + 2
        
        if current_section:
            section_text = '\n\n'.join(current_section)
            section_start = current_pos - current_length
            sections.append((section_text, section_start))
        
        return sections if sections else [(text, 0)]
    
    def _split_into_paragraphs(self, text: str, base_pos: int) -> List[Tuple[str, int]]:
        """
        Split text thành paragraphs
        
        Returns:
            List[Tuple[str, int]]: List of (paragraph_text, start_position)
        """
        # Split theo câu và group thành paragraphs
        sentences = sent_tokenize(text)
        paragraphs = []
        current_para = []
        current_length = 0
        current_pos = base_pos
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_length = len(sent)
            
            if current_length + sent_length + 1 <= self.paragraph_size:
                current_para.append(sent)
                current_length += sent_length + 1
            else:
                if current_para:
                    para_text = ' '.join(current_para)
                    para_start = current_pos
                    paragraphs.append((para_text, para_start))
                    current_pos += len(para_text) + 1
                
                current_para = [sent]
                current_length = sent_length
        
        if current_para:
            para_text = ' '.join(current_para)
            paragraphs.append((para_text, current_pos))
        
        return paragraphs if paragraphs else [(text, base_pos)]
    
    def _split_into_sentences(self, text: str, base_pos: int) -> List[Tuple[str, int]]:
        """
        Split text thành sentences hoặc sentence groups
        
        Returns:
            List[Tuple[str, int]]: List of (sentence_text, start_position)
        """
        sentences = sent_tokenize(text)
        result = []
        current_pos = base_pos
        
        # Group sentences nếu chúng quá ngắn
        current_group = []
        current_length = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_length = len(sent)
            
            # Nếu sentence đơn lẻ đã đủ lớn
            if sent_length >= self.sentence_size:
                # Lưu group hiện tại
                if current_group:
                    group_text = ' '.join(current_group)
                    result.append((group_text, current_pos))
                    current_pos += len(group_text) + 1
                    current_group = []
                    current_length = 0
                
                # Thêm sentence lớn
                result.append((sent, current_pos))
                current_pos += sent_length + 1
            
            # Group các sentences nhỏ
            elif current_length + sent_length + 1 <= self.sentence_size:
                current_group.append(sent)
                current_length += sent_length + 1
            else:
                if current_group:
                    group_text = ' '.join(current_group)
                    result.append((group_text, current_pos))
                    current_pos += len(group_text) + 1
                
                current_group = [sent]
                current_length = sent_length
        
        if current_group:
            group_text = ' '.join(current_group)
            result.append((group_text, current_pos))
        
        return result if result else [(text, base_pos)]
    
    def get_chunks_by_level(
        self, 
        
        chunks: List[HierarchicalChunk], 
        level: ChunkLevel
    ) -> List[HierarchicalChunk]:
        """
        Lấy tất cả chunks ở một level cụ thể
        
        Args:
            chunks: Danh sách tất cả chunks
            level: Level cần lấy
            
        Returns:
            List[HierarchicalChunk]: Chunks ở level đó
        """
        return [chunk for chunk in chunks if chunk.level == level]
    
    def get_chunk_with_context(
        self, 
        chunks: List[HierarchicalChunk], 
        chunk_id: str,
        include_parent: bool = True,
        include_children: bool = False
    ) -> Dict[str, any]:
        """
        Lấy chunk cùng với context (parent và children)
        
        Args:
            chunks: Danh sách tất cả chunks
            chunk_id: ID của chunk cần lấy
            include_parent: Có include parent chunk không
            include_children: Có include children chunks không
            
        Returns:
            Dict chứa chunk và context
        """
        chunk_map = {c.chunk_id: c for c in chunks}
        target_chunk = chunk_map.get(chunk_id)
        
        if not target_chunk:
            return None
        
        result = {
            "chunk": target_chunk,
            "parent": None,
            "children": []
        }
        
        if include_parent and target_chunk.parent_id:
            result["parent"] = chunk_map.get(target_chunk.parent_id)
        
        if include_children:
            result["children"] = [
                chunk_map.get(child_id) 
                for child_id in target_chunk.children_ids 
                if child_id in chunk_map
            ]
        
        return result
    
    def convert_to_documents(
        self, 
        chunks: List[HierarchicalChunk], 
        level: ChunkLevel = ChunkLevel.PARAGRAPH
    ) -> List[Document]:
        """
        Convert hierarchical chunks thành Document objects ở một level cụ thể
        
        Args:
            chunks: Danh sách hierarchical chunks
            level: Level muốn convert (mặc định: PARAGRAPH)
            
        Returns:
            List[Document]: Danh sách documents
        """
        target_chunks = self.get_chunks_by_level(chunks, level)
        documents = []
        
        for chunk in target_chunks:
            metadata = chunk.metadata.copy()
            metadata.update({
                "chunk_id": chunk.chunk_id,
                "level": chunk.level.name,
                "parent_id": chunk.parent_id,
                "children_count": len(chunk.children_ids),
                "start_char": chunk.start_char,
                "end_char": chunk.end_char
            })
            
            doc = Document(content=chunk.content, metadata=metadata)
            documents.append(doc)
        
        return documents
