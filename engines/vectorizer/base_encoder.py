import numpy as np
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

class SessionEncoder(ABC):
    """Abstract base class for session encoding"""
    
    @abstractmethod
    def encode(self, track_ids: List[str]) -> np.ndarray:
        """Encode a sequence of track IDs into a session vector"""
        pass
    
    @abstractmethod
    def load_embeddings(self, embeddings_path: Path) -> None:
        """Load track embeddings from file"""
        pass 