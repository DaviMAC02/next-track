import os
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Centralized configuration management"""

    def __init__(self):
        # API Configuration
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))
        self.API_TITLE: str = os.getenv("API_TITLE", "NextTrack API")
        self.API_VERSION: str = os.getenv("API_VERSION", "1.0.0")

        # Paths
        self.DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
        self.MODELS_DIR: Path = self.DATA_DIR / "models"
        self.EMBEDDINGS_FILE: Path = self.DATA_DIR / os.getenv(
            "EMBEDDINGS_FILE", "lightfm_embeddings.json"
        )

        # Recommendation Parameters
        self.DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "10"))
        self.MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", "50"))

        # Hybrid Weights - Optimized based on FINAL_REPORT.md benchmarks
        self.CF_WEIGHT: float = float(os.getenv("CF_WEIGHT", "0.6"))  # Reduced from 0.7
        self.CB_WEIGHT: float = float(
            os.getenv("CB_WEIGHT", "0.4")
        )  # Increased from 0.3

        # ANN Index Configuration
        self.USE_ANN_INDEX: bool = os.getenv("USE_ANN_INDEX", "true").lower() == "true"
        self.ANN_INDEX_TYPE: str = os.getenv("ANN_INDEX_TYPE", "faiss")  # faiss, annoy
        self.ANN_N_TREES: int = int(os.getenv("ANN_N_TREES", "10"))
        self.ANN_SEARCH_K: int = int(os.getenv("ANN_SEARCH_K", "100"))

        # Security
        self.ENABLE_AUTH: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"
        self.API_KEY: str = os.getenv("API_KEY", "")
        self.RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.RATE_LIMIT_WINDOW: int = int(
            os.getenv("RATE_LIMIT_WINDOW", "3600")
        )  # seconds

        # Logging
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT: str = os.getenv(
            "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Content-Based Configuration
        self.USE_CONTENT_BASED: bool = (
            os.getenv("USE_CONTENT_BASED", "true").lower() == "true"
        )
        self.METADATA_FEATURES_FILE: Path = self.DATA_DIR / os.getenv(
            "METADATA_FEATURES_FILE", "metadata_features.npy"
        )

        # Session Encoder Configuration
        self.SESSION_ENCODER_TYPE: str = os.getenv(
            "SESSION_ENCODER_TYPE", "mean_pooling"
        )  # mean_pooling, lstm, transformer
        self.ENABLE_LEARNED_ENCODERS: bool = (
            os.getenv("ENABLE_LEARNED_ENCODERS", "false").lower() == "true"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_") and not callable(getattr(self, key))
        }


# Global settings instance
settings = Settings()
