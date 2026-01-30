from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# ======================================================
# Load environment
# ======================================================

# Get project root (where .env should be located)
# This resolves to Vi-RAG/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE_FILE = PROJECT_ROOT / ".env.example"

# Try to load .env file
load_dotenv(ENV_FILE)

# ======================================================
# Helper Functions
# ======================================================

def print_setup_instructions():
    """Print helpful instructions for setting up environment variables."""
    print("\n" + "=" * 70)
    print("  VI-RAG CONFIGURATION SETUP")
    print("=" * 70)
    print("\n Missing required API keys!")
    print("\nYou have two options to configure Vi-RAG:\n")
    
    print(" OPTION 1: Create a .env file (Recommended)")
    print("-" * 70)
    
    if ENV_EXAMPLE_FILE.exists():
        print(f"1. Copy the example file:")
        print(f"   cp {ENV_EXAMPLE_FILE} {ENV_FILE}")
        print(f"\n2. Edit {ENV_FILE} and add your API keys:")
    else:
        print(f"1. Create a file named .env in: {PROJECT_ROOT}")
        print(f"\n2. Add the following content to {ENV_FILE}:")
    
    print("""
   GEMINI_API_KEY=your_gemini_api_key_here
   QDRANT_API_KEY=your_qdrant_api_key_here
   QDRANT_URL=your_qdrant_url_here
    """)
    
    print("\n OPTION 2: Set environment variables manually")
    print("-" * 70)
    print("Set the following environment variables in your shell:\n")
    
    if os.name == 'nt':  # Windows
        print("   set GEMINI_API_KEY=your_gemini_api_key_here")
        print("   set QDRANT_API_KEY=your_qdrant_api_key_here")
        print("   set QDRANT_URL=your_qdrant_url_here")
    else:  # Linux/Mac
        print("   export GEMINI_API_KEY=your_gemini_api_key_here")
        print("   export QDRANT_API_KEY=your_qdrant_api_key_here")
        print("   export QDRANT_URL=your_qdrant_url_here")
    
    print("\n" + "=" * 70)
    print(" Get your API keys from:")
    print("=" * 70)
    print("  • Gemini API Key: https://makersuite.google.com/app/apikey")
    print("  • Qdrant Cloud:   https://cloud.qdrant.io")
    print("=" * 70)
    print()


def get_env(name: str, default=None, required: bool = False):
    """
    Get environment variable with optional default and required check.
    
    Args:
        name: Environment variable name
        default: Default value if not found
        required: Whether this variable is required
    
    Returns:
        Environment variable value
    
    Raises:
        RuntimeError: If required variable is missing
    """
    value = os.getenv(name, default)
    
    if required and value is None:
        return None  # Will be handled in validation
    
    return value


def validate_required_settings():
    """
    Validate that all required settings are present.
    Provides helpful error messages if any are missing.
    """
    missing_vars = []
    
    # Check for required variables
    if not os.getenv("GEMINI_API_KEY"):
        missing_vars.append("GEMINI_API_KEY")
    if not os.getenv("QDRANT_API_KEY"):
        missing_vars.append("QDRANT_API_KEY")
    if not os.getenv("QDRANT_URL"):
        missing_vars.append("QDRANT_URL")
    
    if missing_vars:
        print_setup_instructions()
        print(f" Missing required variables: {', '.join(missing_vars)}\n")
        sys.exit(1)


# ======================================================
# Validate Configuration
# ======================================================

# Only validate when not in documentation/testing mode
if __name__ != "__main__" and "sphinx" not in sys.modules:
    validate_required_settings()

# ======================================================
# Core settings
# ======================================================

ENV = get_env("ENV", "development")

GEMINI_API_KEY = get_env("GEMINI_API_KEY")

EMBEDDING_MODEL = get_env(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_TOP_K = int(get_env("VECTOR_TOP_K", 5))

# Embedding Configuration
EMBEDDING_DIM = int(get_env("EMBEDDING_DIM", 768))

# Qdrant Configuration
QDRANT_HOST = get_env("QDRANT_HOST", "localhost")
QDRANT_PORT = int(get_env("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = get_env("QDRANT_COLLECTION_NAME", "rag_documents")
QDRANT_VECTOR_DIM = int(get_env("QDRANT_VECTOR_DIM", 768))
QDRANT_API_KEY = get_env("QDRANT_API_KEY")
QDRANT_URL = get_env("QDRANT_URL")

# ======================================================
# Export
# ======================================================

__all__ = [
    "ENV",
    "GEMINI_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "VECTOR_TOP_K",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_COLLECTION_NAME",
    "QDRANT_VECTOR_DIM",
    "QDRANT_API_KEY",
    "QDRANT_URL",
]
