from typing import List

class SimpleTextSplitter:
    """Class đơn giản để cắt văn bản (thay thế RecursiveCharacterTextSplitter của LangChain)"""
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Cắt văn bản thành các đoạn chồng lấn nhau"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start += (self.chunk_size - self.chunk_overlap)
        return chunks