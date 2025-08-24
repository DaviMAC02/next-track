import numpy as np
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from .base_encoder import SessionEncoder

logger = logging.getLogger(__name__)


class MeanPoolingEncoder(SessionEncoder):
    """Simple mean pooling session encoder - production implementation"""

    def __init__(self):
        self.track_ids: List[str] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self._track_to_idx: Dict[str, int] = {}

    def load_embeddings(self, embeddings_path: Path) -> None:
        """Load embeddings from JSON file"""
        try:
            with open(embeddings_path) as f:
                raw_embeddings = json.load(f)

            self.track_ids = list(raw_embeddings.keys())
            self.embeddings_matrix = np.vstack(
                [raw_embeddings[tid] for tid in self.track_ids]
            )
            self._track_to_idx = {tid: idx for idx, tid in enumerate(self.track_ids)}

            logger.info(
                f"Loaded {len(self.track_ids)} track embeddings from {embeddings_path}"
            )

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    def encode(self, track_ids: List[str]) -> np.ndarray:
        """Encode session with recency-weighted averaging"""
        if self.embeddings_matrix is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings first.")

        if not track_ids:
            raise ValueError("track_ids cannot be empty")

        # Find indices for input tracks
        try:
            indices = [self._track_to_idx[tid] for tid in track_ids]
        except KeyError as e:
            raise ValueError(f"Unknown track ID: {e}")

        # Apply recency weighting - more recent tracks get higher weights
        n_tracks = len(indices)
        weights = np.array([np.exp(i / n_tracks) for i in range(n_tracks)])
        weights = weights / weights.sum()  # Normalize

        # Weighted average instead of simple mean
        track_embeddings = self.embeddings_matrix[indices]
        session_vector = np.average(track_embeddings, axis=0, weights=weights)

        return session_vector.reshape(1, -1)


class SessionVectorizer:
    """Main vectorizer class - simplified for production use"""

    def __init__(
        self, encoder_type: str = "mean_pooling", model_path: Optional[Path] = None
    ):
        """Initialize with mean pooling encoder only"""
        self.encoder_type = encoder_type
        self.encoder = MeanPoolingEncoder()

    def load_embeddings(self, embeddings_path: Path) -> None:
        """Load embeddings into the encoder"""
        self.encoder.load_embeddings(embeddings_path)

    def encode_session(self, track_ids: List[str]) -> np.ndarray:
        """Encode a session into a vector"""
        return self.encoder.encode(track_ids)

    def get_track_ids(self) -> List[str]:
        """Get list of available track IDs"""
        return self.encoder.track_ids

    def get_embeddings_matrix(self) -> np.ndarray:
        """Get the full embeddings matrix"""
        if self.encoder.embeddings_matrix is None:
            raise ValueError("Embeddings not loaded")
        return self.encoder.embeddings_matrix
