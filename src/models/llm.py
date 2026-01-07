"""
Module quản lý Large Language Models (LLM)
Sử dụng Google Generative AI SDK trực tiếp (không dùng LangChain)
Tích hợp PyTorch cho tensor operations
"""

import os
from typing import Optional, Dict, Any, List, Generator, Union
from google import genai
from google.genai import types
import torch
from src.core.config import settings
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


class GeminiLLM:
    """
    Wrapper cho Google Gemini LLM sử dụng google-generativeai SDK
    Hỗ trợ: gemini-1.5-flash, gemini-1.5-pro, gemini-pro
    Tích hợp PyTorch cho tensor operations
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 40,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Khởi tạo Gemini LLM
        
        Args:
            model_name: Tên model (mặc định từ config)
            temperature: Độ sáng tạo (0-1)
            max_output_tokens: Số token tối đa trong response
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            device: PyTorch device ('cuda', 'cpu', hoặc None để auto-detect)
        """
        self.model_name = model_name or settings.LLM_MODEL_NAME
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        
        # Setup PyTorch device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Kiểm tra API key
        self.api_key = settings.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. "
                "Please set it in .env file or environment variables."
            )
        
        # Configure và khởi tạo model
        self.client = genai.Client(api_key=self.api_key)
        
        # Generation config
        self.generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
        )
    
    def invoke(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Gọi LLM với prompt đơn giản
        
        Args:
            prompt: Câu hỏi/yêu cầu
            system_instruction: System instruction (optional)
            **kwargs: Các tham số bổ sung
            
        Returns:
            Response text từ LLM
        """
        # Merge system instruction vào prompt nếu có
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Generate response
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=self.generation_config
        )
        
        return response.text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Chat với LLM sử dụng conversation history
        
        Args:
            messages: List of messages [{"role": "user/assistant/system", "content": "..."}]
            **kwargs: Các tham số bổ sung
            
        Returns:
            Response text từ LLM
        """
        # Extract system instruction nếu có
        system_instruction = None
        chat_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                chat_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant" or role == "model":
                chat_messages.append({"role": "model", "parts": [content]})
        
        # Convert to new format
        history = []
        for msg in chat_messages[:-1]:
            history.append(types.Content(
                role=msg["role"],
                parts=[types.Part(text=msg["parts"][0])]
            ))
        
        # Generate response with history
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=history + [types.Content(
                role="user",
                parts=[types.Part(text=chat_messages[-1]["parts"][0])]
            )],
            config=self.generation_config
        )
        
        return response.text
    
    def stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream response từ LLM (real-time output)
        
        Args:
            prompt: Câu hỏi/yêu cầu
            system_instruction: System instruction (optional)
            **kwargs: Các tham số bổ sung
            
        Yields:
            Response chunks
        """
        # Merge system instruction vào prompt nếu có
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Generate streaming response
        response = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=full_prompt,
            config=self.generation_config
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def generate_with_context(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate answer với context (cho RAG)
        
        Args:
            question: Câu hỏi của user
            context: Context/documents từ retrieval
            system_prompt: Custom system prompt (optional)
            **kwargs: Các tham số bổ sung
            
        Returns:
            Generated answer
        """
        default_system_prompt = """Bạn là một trợ lý AI thông minh và hữu ích.
Hãy trả lời câu hỏi dựa trên context được cung cấp.
Nếu context không chứa thông tin liên quan, hãy nói rằng bạn không có đủ thông tin để trả lời."""
        
        system = system_prompt or default_system_prompt
        
        # Build prompt
        full_prompt = f"""{system}

Context: {context}

Question: {question}

Answer:"""
        
        return self.invoke(full_prompt, **kwargs)
    
    def embed_text(
        self,
        text: str,
        task_type: str = "retrieval_document"
    ) -> torch.Tensor:
        """
        Generate embeddings cho text sử dụng Gemini Embedding API
        Trả về PyTorch tensor
        
        Args:
            text: Text cần embed
            task_type: Task type ('retrieval_document', 'retrieval_query', 'semantic_similarity')
            
        Returns:
            PyTorch tensor của embedding
        """
        # Use the new embedding API
        result = self.client.models.embed_content(
            model="models/text-embedding-004",
            contents=text
        )
        
        # Convert to PyTorch tensor
        embedding = torch.tensor(result.embeddings[0].values, device=self.device)
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        task_type: str = "retrieval_document"
    ) -> torch.Tensor:
        """
        Generate embeddings cho batch texts
        
        Args:
            texts: List of texts
            task_type: Task type
            
        Returns:
            PyTorch tensor shape (batch_size, embedding_dim)
        """
        embeddings = []
        for text in texts:
            emb = self.embed_text(text, task_type)
            embeddings.append(emb)
        
        # Stack thành batch tensor
        return torch.stack(embeddings)
    
    def compute_similarity(
        self,
        query_embedding: torch.Tensor,
        doc_embeddings: torch.Tensor,
        metric: str = "cosine"
    ) -> torch.Tensor:
        """
        Tính similarity giữa query và documents sử dụng PyTorch
        
        Args:
            query_embedding: Query embedding tensor (embedding_dim,)
            doc_embeddings: Document embeddings tensor (num_docs, embedding_dim)
            metric: 'cosine' hoặc 'dot'
            
        Returns:
            Similarity scores tensor (num_docs,)
        """
        if metric == "cosine":
            # Cosine similarity
            query_norm = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
            doc_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
            similarities = torch.mm(query_norm, doc_norm.t()).squeeze()
        elif metric == "dot":
            # Dot product
            similarities = torch.mv(doc_embeddings, query_embedding)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarities
    
    def count_tokens(self, text: str) -> int:
        """
        Đếm số tokens trong text
        
        Args:
            text: Text cần đếm
            
        Returns:
            Số tokens
        """
        result = self.client.models.count_tokens(
            model=self.model_name,
            contents=text
        )
        return result.total_tokens
    
    def get_model_info(self) -> Dict[str, Any]:
        """Lấy thông tin về model"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "device": str(self.device),
            "provider": "Google Gemini (Native SDK)",
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }


class LLMFactory:
    """Factory class để tạo các LLM khác nhau"""
    
    @staticmethod
    def create_llm(
        provider: str = "gemini",
        model_name: Optional[str] = None,
        **kwargs
    ) -> GeminiLLM:
        """
        Tạo LLM instance
        
        Args:
            provider: "gemini" (có thể extend cho providers khác)
            model_name: Tên model (optional)
            **kwargs: Các tham số cho LLM
            
        Returns:
            LLM instance
        """
        if provider.lower() == "gemini":
            return GeminiLLM(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# Convenience function
def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    device: Optional[str] = None,
    **kwargs
) -> GeminiLLM:
    """
    Quick way to get LLM instance
    
    Usage:
        llm = get_llm()
        response = llm.invoke("Hello, how are you?")
        
        # With PyTorch GPU
        llm = get_llm(device='cuda')
        embedding = llm.embed_text("Some text")
    """
    return GeminiLLM(
        model_name=model_name,
        temperature=temperature,
        device=device,
        **kwargs
    )
