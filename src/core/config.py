import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load biến môi trường từ .env
load_dotenv()

class AppConfig:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
        self.CONFIG_PATH = self.BASE_DIR / "config.yaml"
        self._config = self._load_yaml()

    def _load_yaml(self):
        if not self.CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found at {self.CONFIG_PATH}")
        with open(self.CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def DATA_DIR(self):
        return self.BASE_DIR / "data"
    
    @property
    def CHROMA_DB_DIR(self):
        return self.DATA_DIR / "indexes" / "chroma"

    # --- Lấy cấu hình từ YAML ---
    @property
    def EMBEDDING_MODEL_NAME(self):
        return self._config.get("embedding_model", "BAAI/bge-m3")

    @property
    def LLM_MODEL_NAME(self):
        return self._config.get("llm_model", "gemini-pro")
    
    @property
    def CHUNKING_CONFIG(self):
        return self._config.get("chunking", {})

    # --- Lấy API KEY từ ENV (An toàn hơn) ---
    @property
    def GOOGLE_API_KEY(self):
        return os.getenv("GOOGLE_API_KEY")

    @property
    def OPENAI_API_KEY(self):
        return os.getenv("OPENAI_API_KEY")

# Singleton instance
settings = AppConfig()