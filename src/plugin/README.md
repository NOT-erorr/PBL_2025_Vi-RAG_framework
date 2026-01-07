# Plugin Directory
    
This directory contains plugin implementations for the RAG system.

## Available Plugins

### 1. FaissVectorStore.py
FAISS-based vector store implementation for efficient similarity search.

**Key Features:**
- Fast similarity search using FAISS library
- Support for different index types (Flat, IVF, HNSW)
- Persistence (save/load)
- Metadata filtering

**Usage:**
```python
from plugin.FaissVectorStore import FaissVectorStore

vector_store = FaissVectorStore(embedding_dimension=1536)
vector_store.add_texts(texts, embeddings, metadatas, ids)
results = vector_store.similarity_search(query_embedding, k=5)
```

### 2. ParentDocumentStore.py
Parent Document Retrieval Strategy implementation for advanced RAG.

**Key Features:**
- Parent-child chunking for better context
- Separate indexing and retrieval chunks
- Storage and mapping management
- JSON persistence

**Components:**
- `ParentChildChunk`: Dataclass for parent-child relationships
- `ParentDocumentStore`: Storage system for parent chunks
- `ParentChildTextSplitter`: Splitter creating parent-child chunks

**Usage:**
```python
from plugin.ParentDocumentStore import (
    ParentChildTextSplitter,
    ParentDocumentStore
)

# Split documents
splitter = ParentChildTextSplitter(
    parent_chunk_size=2000,
    child_chunk_size=400
)
chunks = splitter.split_text(text)

# Store parent chunks
store = ParentDocumentStore()
store.add_chunks(chunks)
store.save("parent_store.json")

# Retrieve parent from child ID
parent = store.get_parent_by_child_id(child_id)
```

## Plugin Development Guidelines

### Creating a New Plugin

1. **Create a new Python file** in `src/plugin/`
2. **Import required dependencies** from core modules
3. **Implement the interface** (if applicable)
4. **Add comprehensive docstrings**
5. **Include usage examples**

### Example Plugin Structure

```python
"""
Plugin Name

Brief description of what this plugin does.

Author: Your Name
Date: YYYY-MM-DD
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class YourPlugin:
    """
    Main plugin class
    
    Args:
        param1: Description
        param2: Description
    
    Example:
        >>> plugin = YourPlugin(param1="value")
        >>> result = plugin.method()
    """
    
    def __init__(self, param1: str, param2: int):
        self.param1 = param1
        self.param2 = param2
        logger.info(f"Initialized YourPlugin with {param1}")
    
    def method(self) -> str:
        """
        Method description
        
        Returns:
            str: Result description
        """
        # Implementation
        pass
```

### Best Practices

1. **Logging**: Use the logging module for debugging and info messages
2. **Error Handling**: Handle exceptions gracefully with informative messages
3. **Type Hints**: Use type hints for all function parameters and returns
4. **Documentation**: Provide clear docstrings with examples
5. **Testing**: Create test files in `testing/code/` directory
6. **Dependencies**: Minimize external dependencies, use core modules when possible

## Testing Plugins

Each plugin should have corresponding test files in `testing/code/`:

```
testing/
├── code/
│   ├── vectorstore/
│   │   └── test_vectorstore.py
│   └── splitter/
│       ├── test_parent_plugin.py
│       └── parent_chunking_demo.py
```

### Running Tests

```bash
# Test individual plugin
python testing/code/splitter/test_parent_plugin.py

# Run comprehensive demo
python testing/code/splitter/parent_chunking_demo.py
```

## Integration with Core

Plugins extend the core functionality without modifying core modules:

```
src/
├── core/              # Core functionality (stable)
│   ├── loader.py
│   ├── splitter.py
│   ├── embedding.py
│   └── vectorstore.py
└── plugin/            # Extensions (flexible)
    ├── FaissVectorStore.py
    └── ParentDocumentStore.py
```

**Import Pattern:**
```python
# Core modules
from core.loader import TXTLoader
from core.splitter import VietnameseTextSplitter

# Plugin modules
from plugin.FaissVectorStore import FaissVectorStore
from plugin.ParentDocumentStore import ParentChildTextSplitter
```

## Contributing

When adding a new plugin:

1. **Follow the structure** outlined above
2. **Add documentation** to this README
3. **Create test files** demonstrating usage
4. **Update examples** if needed
5. **Consider backward compatibility**

---

**For detailed documentation on specific plugins, see:**
- [Parent Chunking Guide](../../docs/PARENT_CHUNKING_GUIDE.md)
- Core module documentation in respective files
