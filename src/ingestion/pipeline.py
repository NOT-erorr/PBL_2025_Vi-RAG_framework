"""
RAG Pipeline - Tổng hợp toàn bộ workflow
Tích hợp: Loading -> Splitting -> Embedding -> Indexing -> Retrieval -> Generation
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json
import torch
from tqdm import tqdm

# Import các components
from src.ingestion.loader import DocumentLoader, Document
from src.ingestion.splitter import ParentDocumentSplitter, RecursiveCharacterSplitter
from src.models.embedding import BGEM3Embedding
from src.models.llm import get_llm, GeminiLLM
from src.plugin.FaissVectorStore import FaissVectorStore
from src.plugin.ParentDocumentStore import ParentDocumentStore
from src.retrieval.retriever import HierarchicalRetriever
from src.core.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration cho RAG Pipeline"""
    # Chunking config
    parent_chunk_size: int = 1500
    child_chunk_size: int = 400
    chunk_overlap: int = 50
    
    # Retrieval config
    top_k: int = 5
    use_rerank: bool = False
    
    # Generation config
    temperature: float = 0.3
    max_output_tokens: int = 2048
    
    # Storage paths
    vectorstore_path: Optional[str] = None
    parent_store_path: Optional[str] = None
    
    # Device
    device: str = "cpu"


class RAGPipeline:
    """
    Complete RAG Pipeline
    Xử lý end-to-end: từ documents đến generated answers
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        embedding_model: Optional[BGEM3Embedding] = None,
        llm: Optional[GeminiLLM] = None
    ):
        """
        Initialize RAG Pipeline
        
        Args:
            config: Pipeline configuration
            embedding_model: Custom embedding model (optional)
            llm: Custom LLM (optional)
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        logger.info("Initializing RAG Pipeline components...")
        
        # Document loader
        self.loader = DocumentLoader()
        
        # Text splitter
        self.splitter = ParentDocumentSplitter(
            parent_splitter=RecursiveCharacterSplitter(
                chunk_size=self.config.parent_chunk_size,
                chunk_overlap=self.config.chunk_overlap
            ),
            child_splitter=RecursiveCharacterSplitter(
                chunk_size=self.config.child_chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        )
        
        # Embedding model
        if embedding_model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
            self.embedding = BGEM3Embedding(
                model_name=settings.EMBEDDING_MODEL_NAME,
                device=self.config.device
            )
        else:
            self.embedding = embedding_model
        
        # LLM
        if llm is None:
            logger.info(f"Loading LLM: {settings.LLM_MODEL_NAME}")
            self.llm = get_llm(
                model_name=settings.LLM_MODEL_NAME,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                device=self.config.device
            )
        else:
            self.llm = llm
        
        # Vector store và parent store
        self.vectorstore = None
        self.parent_store = None
        self.retriever = None
        
        logger.info("✅ RAG Pipeline initialized successfully")
    
    def ingest_documents(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest documents vào pipeline
        
        Args:
            file_paths: Path(s) to document(s)
            show_progress: Show progress bar
            
        Returns:
            Dict với statistics
        """
        logger.info("=" * 60)
        logger.info("STEP 1: DOCUMENT LOADING")
        logger.info("=" * 60)
        
        # Convert to list
        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        
        # Load documents
        all_docs = []
        for file_path in tqdm(file_paths, desc="Loading documents", disable=not show_progress):
            docs = self.loader.load(file_path)
            all_docs.extend(docs)
        
        logger.info(f"✅ Loaded {len(all_docs)} documents")
        
        # Split documents
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: TEXT SPLITTING")
        logger.info("=" * 60)
        
        parent_chunks = []
        child_chunks = []
        
        for doc in tqdm(all_docs, desc="Splitting documents", disable=not show_progress):
            result = self.splitter.split_document(doc)
            parent_chunks.extend(result["parent_chunks"])
            child_chunks.extend(result["child_chunks"])
        
        logger.info(f"✅ Created {len(parent_chunks)} parent chunks")
        logger.info(f"✅ Created {len(child_chunks)} child chunks")
        
        # Generate embeddings
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: EMBEDDING GENERATION")
        logger.info("=" * 60)
        
        # Embed child chunks (for retrieval)
        child_texts = [chunk.content for chunk in child_chunks]
        logger.info(f"Embedding {len(child_texts)} child chunks...")
        
        child_embeddings = []
        batch_size = 32
        for i in tqdm(range(0, len(child_texts), batch_size), desc="Embedding", disable=not show_progress):
            batch = child_texts[i:i + batch_size]
            batch_embs = self.embedding.embed_documents(batch)
            
            # Handle different embedding types
            if isinstance(batch_embs, dict):
                # Multi-vector embeddings (BGE-M3)
                batch_embs = batch_embs.get('dense_vecs', batch_embs.get('colbert_vecs', []))
            
            child_embeddings.extend(batch_embs)
        
        logger.info(f"✅ Generated {len(child_embeddings)} embeddings")
        
        # Create vector store
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: INDEXING")
        logger.info("=" * 60)
        
        # Initialize FAISS vector store
        self.vectorstore = FaissVectorStore(
            dimension=self.embedding.dimension,
            index_type="Flat"
        )
        
        # Add vectors
        logger.info("Building FAISS index...")
        for i, (chunk, embedding) in enumerate(zip(child_chunks, child_embeddings)):
            metadata = {
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "level": chunk.level.value if hasattr(chunk.level, 'value') else str(chunk.level),
                "source": chunk.metadata.get("source", "unknown")
            }
            self.vectorstore.add_vector(
                vector_id=chunk.chunk_id,
                vector=embedding,
                metadata=metadata
            )
        
        logger.info(f"✅ Indexed {self.vectorstore.get_stats()['total_vectors']} vectors")
        
        # Create parent document store
        logger.info("Building parent document store...")
        self.parent_store = ParentDocumentStore()
        
        for parent in parent_chunks:
            # Find children
            children = [c for c in child_chunks if c.parent_id == parent.chunk_id]
            child_ids = [c.chunk_id for c in children]
            
            self.parent_store.add_parent(
                parent_id=parent.chunk_id,
                parent_content=parent.content,
                child_ids=child_ids,
                metadata=parent.metadata
            )
        
        logger.info(f"✅ Stored {len(self.parent_store.parents)} parent documents")
        
        # Save stores if paths provided
        if self.config.vectorstore_path:
            self.save_vectorstore(self.config.vectorstore_path)
        if self.config.parent_store_path:
            self.save_parent_store(self.config.parent_store_path)
        
        # Initialize retriever
        self.retriever = HierarchicalRetriever(
            vectorstore=self.vectorstore,
            parent_store=self.parent_store,
            embedding_model=self.embedding
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ INGESTION COMPLETE")
        logger.info("=" * 60)
        
        return {
            "num_documents": len(all_docs),
            "num_parent_chunks": len(parent_chunks),
            "num_child_chunks": len(child_chunks),
            "num_vectors": self.vectorstore.get_stats()['total_vectors']
        }
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            return_sources: Include source documents in response
            stream: Stream the response
            
        Returns:
            Dict with answer and metadata
        """
        if self.retriever is None:
            raise ValueError("Pipeline not ready. Please run ingest_documents() first.")
        
        top_k = top_k or self.config.top_k
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: {question}")
        logger.info(f"{'='*60}")
        
        # Retrieve relevant documents
        logger.info(f"Retrieving top {top_k} documents...")
        retrieved_docs = self.retriever.retrieve(
            query=question,
            k=top_k,
            strategy="small_to_big"
        )
        
        logger.info(f"✅ Retrieved {len(retrieved_docs)} documents")
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            # Use parent content if available
            content = doc.parent_content if doc.parent_content else doc.content
            context_parts.append(f"[Document {i}]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        logger.info("Generating answer...")
        
        if stream:
            # Streaming generation
            logger.info("Streaming response:")
            print("\nAnswer: ", end="", flush=True)
            
            full_answer = ""
            for chunk in self.llm.stream(
                prompt=question,
                system_instruction=self._build_system_prompt(context)
            ):
                print(chunk, end="", flush=True)
                full_answer += chunk
            
            print("\n")
            answer = full_answer
        else:
            # Regular generation
            answer = self.llm.generate_with_context(
                question=question,
                context=context,
                system_prompt=self._build_rag_system_prompt()
            )
            logger.info(f"✅ Generated answer ({len(answer)} chars)")
        
        # Prepare response
        response = {
            "question": question,
            "answer": answer,
            "num_sources": len(retrieved_docs)
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "content": doc.content,
                    "score": doc.score,
                    "metadata": doc.metadata,
                    "parent_content": doc.parent_content
                }
                for doc in retrieved_docs
            ]
        
        return response
    
    def batch_query(
        self,
        questions: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of questions
            show_progress: Show progress bar
            
        Returns:
            List of responses
        """
        results = []
        for question in tqdm(questions, desc="Processing queries", disable=not show_progress):
            result = self.query(question, return_sources=False)
            results.append(result)
        
        return results
    
    def save_vectorstore(self, path: Union[str, Path]) -> None:
        """Save vector store to disk"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save(str(path))
        logger.info(f"✅ Vector store saved to {path}")
    
    def load_vectorstore(self, path: Union[str, Path]) -> None:
        """Load vector store from disk"""
        self.vectorstore = FaissVectorStore.load(str(path))
        logger.info(f"✅ Vector store loaded from {path}")
    
    def save_parent_store(self, path: Union[str, Path]) -> None:
        """Save parent document store to disk"""
        if self.parent_store is None:
            raise ValueError("Parent store not initialized")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.parent_store.save(str(path))
        logger.info(f"✅ Parent store saved to {path}")
    
    def load_parent_store(self, path: Union[str, Path]) -> None:
        """Load parent document store from disk"""
        self.parent_store = ParentDocumentStore.load(str(path))
        logger.info(f"✅ Parent store loaded from {path}")
    
    def load_from_disk(
        self,
        vectorstore_path: Union[str, Path],
        parent_store_path: Union[str, Path]
    ) -> None:
        """
        Load complete pipeline from disk
        
        Args:
            vectorstore_path: Path to vector store
            parent_store_path: Path to parent store
        """
        logger.info("Loading pipeline from disk...")
        
        self.load_vectorstore(vectorstore_path)
        self.load_parent_store(parent_store_path)
        
        # Initialize retriever
        self.retriever = HierarchicalRetriever(
            vectorstore=self.vectorstore,
            parent_store=self.parent_store,
            embedding_model=self.embedding
        )
        
        logger.info("✅ Pipeline loaded successfully")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            "embedding_model": self.embedding.model_name,
            "llm_model": self.llm.model_name if hasattr(self.llm, 'model_name') else "unknown",
            "device": self.config.device
        }
        
        if self.vectorstore:
            stats.update({
                "vectorstore": self.vectorstore.get_stats()
            })
        
        if self.parent_store:
            stats.update({
                "num_parents": len(self.parent_store.parents)
            })
        
        return stats
    
    def _build_rag_system_prompt(self) -> str:
        """Build system prompt for RAG"""
        return """Bạn là một trợ lý AI thông minh và hữu ích.
Nhiệm vụ của bạn là trả lời câu hỏi dựa trên context được cung cấp từ documents.

Hướng dẫn:
1. Đọc kỹ context được cung cấp
2. Trả lời câu hỏi một cách chính xác và đầy đủ
3. Nếu context không chứa thông tin liên quan, hãy nói rõ
4. Trích dẫn thông tin từ context khi có thể
5. Trả lời bằng tiếng Việt một cách tự nhiên và dễ hiểu"""
    
    def _build_system_prompt(self, context: str) -> str:
        """Build system prompt with context"""
        return f"""{self._build_rag_system_prompt()}

Context từ documents:
{context}"""


def create_pipeline(
    config: Optional[PipelineConfig] = None,
    device: Optional[str] = None
) -> RAGPipeline:
    """
    Convenience function để tạo pipeline
    
    Args:
        config: Pipeline configuration
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        RAGPipeline instance
    """
    if config is None:
        config = PipelineConfig()
    
    if device:
        config.device = device
    
    return RAGPipeline(config=config)


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = create_pipeline()
    
    # Ingest documents
    docs_path = "path/to/documents"
    stats = pipeline.ingest_documents(docs_path)
    print(f"\nIngestion stats: {stats}")
    
    # Query
    question = "What is RAG?"
    response = pipeline.query(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {response['answer']}")
