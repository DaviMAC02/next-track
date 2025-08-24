import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SimilarityIndex(ABC):
    """Abstract base class for similarity search"""
    
    @abstractmethod
    def build_index(self, embeddings: np.ndarray) -> None:
        """Build the similarity index"""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int, exclude_indices: List[int] = None) -> Tuple[List[int], List[float]]:
        """Search for k most similar items"""
        pass

class BruteForceCosineIndex(SimilarityIndex):
    """Brute force cosine similarity search - current implementation"""
    
    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """Store embeddings matrix"""
        self.embeddings = embeddings
        logger.info(f"Built brute force index with {embeddings.shape[0]} items")
    
    def search(self, query_vector: np.ndarray, k: int, exclude_indices: List[int] = None) -> Tuple[List[int], List[float]]:
        """Compute cosine similarities and return top-k"""
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.embeddings).flatten()
        
        # Exclude specified indices
        if exclude_indices:
            for idx in exclude_indices:
                if 0 <= idx < len(similarities):
                    similarities[idx] = -1.0
        
        # Get top-k indices
        top_indices = similarities.argsort()[::-1][:k]
        top_scores = similarities[top_indices]
        
        return top_indices.tolist(), top_scores.tolist()

class FAISSIndex(SimilarityIndex):
    """FAISS-based approximate nearest neighbor search"""
    
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.index = None
        self.embeddings: Optional[np.ndarray] = None
        
        try:
            import faiss
            self.faiss = faiss
            logger.info("FAISS available for ANN search")
        except ImportError:
            logger.warning("FAISS not available, falling back to brute force search")
            self.faiss = None
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS index"""
        if self.faiss is None:
            raise ImportError("FAISS not available")
        
        self.embeddings = embeddings.astype(np.float32)
        dimension = embeddings.shape[1]
        
        if self.metric == "cosine":
            # Normalize vectors for cosine similarity
            self.faiss.normalize_L2(self.embeddings)
            self.index = self.faiss.IndexFlatIP(dimension)  # Inner product after normalization = cosine
        else:
            self.index = self.faiss.IndexFlatL2(dimension)  # L2 distance
        
        self.index.add(self.embeddings)
        logger.info(f"Built FAISS index with {embeddings.shape[0]} items")
    
    def search(self, query_vector: np.ndarray, k: int, exclude_indices: List[int] = None) -> Tuple[List[int], List[float]]:
        """Search using FAISS index"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        query = query_vector.astype(np.float32)
        if self.metric == "cosine":
            self.faiss.normalize_L2(query)
        
        # Search for more results to account for exclusions
        search_k = k + len(exclude_indices or [])
        search_k = min(search_k, self.index.ntotal)
        
        scores, indices = self.index.search(query, search_k)
        
        # Filter out excluded indices
        filtered_indices = []
        filtered_scores = []
        exclude_set = set(exclude_indices or [])
        
        for idx, score in zip(indices[0], scores[0]):
            if idx not in exclude_set and len(filtered_indices) < k:
                filtered_indices.append(int(idx))
                filtered_scores.append(float(score))
        
        return filtered_indices, filtered_scores

class AnnoyIndex(SimilarityIndex):
    """Annoy-based approximate nearest neighbor search"""
    
    def __init__(self, n_trees: int = 10, metric: str = "angular"):
        self.n_trees = n_trees
        self.metric = metric  # angular (cosine), euclidean
        self.index = None
        self.dimension = None
        
        try:
            from annoy import AnnoyIndex
            self.annoy_cls = AnnoyIndex
            logger.info("Annoy available for ANN search")
        except ImportError:
            logger.warning("Annoy not available, falling back to brute force search")
            self.annoy_cls = None
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """Build Annoy index"""
        if self.annoy_cls is None:
            raise ImportError("Annoy not available")
        
        self.dimension = embeddings.shape[1]
        self.index = self.annoy_cls(self.dimension, self.metric)
        
        for i, vector in enumerate(embeddings):
            self.index.add_item(i, vector)
        
        self.index.build(self.n_trees)
        logger.info(f"Built Annoy index with {embeddings.shape[0]} items, {self.n_trees} trees")
    
    def search(self, query_vector: np.ndarray, k: int, exclude_indices: List[int] = None) -> Tuple[List[int], List[float]]:
        """Search using Annoy index"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Search for more results to account for exclusions
        search_k = k + len(exclude_indices or [])
        search_k = min(search_k, self.index.get_n_items())
        
        indices, scores = self.index.get_nns_by_vector(query_vector.flatten(), search_k, include_distances=True)
        
        # Filter out excluded indices
        filtered_indices = []
        filtered_scores = []
        exclude_set = set(exclude_indices or [])
        
        for idx, score in zip(indices, scores):
            if idx not in exclude_set and len(filtered_indices) < k:
                filtered_indices.append(int(idx))
                # Convert distance to similarity (smaller distance = higher similarity)
                similarity = 1.0 / (1.0 + score) if self.metric == "euclidean" else 1.0 - score
                filtered_scores.append(float(similarity))
        
        return filtered_indices, filtered_scores

class CollaborativeFilteringEngine:
    """Collaborative filtering recommendation engine"""
    
    def __init__(self, index_type: str = "brute_force", **index_kwargs):
        self.index_type = index_type
        
        # Create appropriate index
        if index_type == "brute_force":
            self.index = BruteForceCosineIndex()
        elif index_type == "faiss":
            self.index = FAISSIndex(**index_kwargs)
        elif index_type == "annoy":
            self.index = AnnoyIndex(**index_kwargs)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.track_ids: List[str] = []
    
    def build_index(self, embeddings: np.ndarray, track_ids: List[str]) -> None:
        """Build the similarity index"""
        self.index.build_index(embeddings)
        self.track_ids = track_ids
        logger.info(f"CF engine built with {len(track_ids)} tracks using {self.index_type} index")
    
    def get_recommendations(self, session_vector: np.ndarray, k: int, exclude_track_ids: List[str] = None) -> Tuple[List[str], List[float]]:
        """Get top-k recommendations based on session vector"""
        # Convert track IDs to indices for exclusion
        exclude_indices = []
        if exclude_track_ids:
            track_to_idx = {tid: idx for idx, tid in enumerate(self.track_ids)}
            exclude_indices = [track_to_idx[tid] for tid in exclude_track_ids if tid in track_to_idx]
        
        # Search for similar tracks
        indices, scores = self.index.search(session_vector, k, exclude_indices)
        
        # Convert indices back to track IDs
        recommended_tracks = [self.track_ids[idx] for idx in indices]
        
        return recommended_tracks, scores
    
    def get_track_similarities(self, track_id: str, k: int = 10) -> Tuple[List[str], List[float]]:
        """Get tracks similar to a specific track"""
        if track_id not in self.track_ids:
            raise ValueError(f"Unknown track ID: {track_id}")
        
        track_idx = self.track_ids.index(track_id)
        
        # Use the track's embedding as query
        if hasattr(self.index, 'embeddings') and self.index.embeddings is not None:
            track_vector = self.index.embeddings[track_idx:track_idx+1]
            return self.get_recommendations(track_vector, k, exclude_track_ids=[track_id])
        else:
            raise ValueError("Index does not support track similarity queries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "index_type": self.index_type,
            "num_tracks": len(self.track_ids),
            "track_ids_sample": self.track_ids[:5] if self.track_ids else []
        }
