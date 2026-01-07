from src.core.utils import get_logger

logger = get_logger("ingestion")

# Placeholder cho việc kiểm tra thư viện tiếng Việt
try:
    import underthesea
    VIETNAMESE_AVAILABLE = True
    logger.info("Underthesea library detected. Vietnamese processing ready.")
except ImportError:
    VIETNAMESE_AVAILABLE = False
    logger.warning("Underthesea not found. Vietnamese tokenization will be limited.")