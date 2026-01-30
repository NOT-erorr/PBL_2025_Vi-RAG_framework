"""
Test script to demonstrate the improved settings.py error handling.

This script shows how the new settings.py provides helpful error messages
when API keys are missing.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("Testing Vi-RAG Configuration Setup...")
print("=" * 70)

# Test 1: With .env file (should work)
print("\nTest 1: Normal operation (with .env file)")
print("-" * 70)
try:
    from vi_rag.config import settings
    print("✓ Settings loaded successfully!")
    print(f"  - GEMINI_API_KEY: {settings.GEMINI_API_KEY[:20] if settings.GEMINI_API_KEY else 'None'}...")
    print(f"  - QDRANT_URL: {settings.QDRANT_URL[:30] if settings.QDRANT_URL else 'None'}...")
    print(f"  - Configuration loaded from: {Path(__file__).parent.parent / '.env'}")
except SystemExit as e:
    print("❌ Configuration failed (as expected if .env is missing)")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("Configuration test complete!")
print("=" * 70)
