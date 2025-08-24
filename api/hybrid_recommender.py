import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path

from engines.cf_engine.collaborative_filter import CollaborativeFilteringEngine
from engines.cb_engine.content_based import ContentBasedEngine
from engines.vectorizer.session_encoder import SessionVectorizer
from config.settings import settings

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Hybrid recommendation system combining CF and CB approaches"""

    def __init__(self, cf_weight: float = 0.7, cb_weight: float = 0.3):
        """
        Initialize hybrid recommender

        Args:
            cf_weight: Weight for collaborative filtering scores
            cb_weight: Weight for content-based scores
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight

        # Validate weights
        if abs(cf_weight + cb_weight - 1.0) > 1e-6:
            logger.warning(
                f"CF and CB weights don't sum to 1.0: {cf_weight} + {cb_weight} = {cf_weight + cb_weight}"
            )

        # Initialize components
        self.vectorizer: Optional[SessionVectorizer] = None
        self.cf_engine: Optional[CollaborativeFilteringEngine] = None
        self.cb_engine: Optional[ContentBasedEngine] = None

        self._initialized = False

    def initialize(
        self,
        embeddings_path: Path,
        cf_index_type: str = "brute_force",
        use_content_based: bool = True,
        content_features_path: Optional[Path] = None,
    ) -> None:
        """Initialize all recommendation components"""

        try:
            # Initialize session vectorizer with configuration
            encoder_type = "mean_pooling"  # Default
            if settings.ENABLE_LEARNED_ENCODERS:
                encoder_type = settings.SESSION_ENCODER_TYPE

            self.vectorizer = SessionVectorizer(encoder_type=encoder_type)
            self.vectorizer.load_embeddings(embeddings_path)

            # Initialize CF engine
            cf_kwargs = {}
            if cf_index_type == "faiss":
                cf_kwargs = {"metric": "cosine"}
            elif cf_index_type == "annoy":
                cf_kwargs = {"n_trees": settings.ANN_N_TREES, "metric": "angular"}

            self.cf_engine = CollaborativeFilteringEngine(cf_index_type, **cf_kwargs)
            self.cf_engine.build_index(
                self.vectorizer.get_embeddings_matrix(), self.vectorizer.get_track_ids()
            )

            # Initialize CB engine if enabled
            if (
                use_content_based
                and content_features_path
                and content_features_path.exists()
            ):
                self.cb_engine = ContentBasedEngine()
                self.cb_engine.load_features(
                    content_features_path, self.vectorizer.get_track_ids()
                )
                self.cb_engine.build_index(normalize_features=True)
                logger.info("Content-based engine initialized")
            elif use_content_based:
                logger.warning(
                    "Content-based engine requested but features not available"
                )

            self._initialized = True
            logger.info("Hybrid recommender initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize hybrid recommender: {e}")
            raise

    def get_recommendations(
        self,
        track_sequence: List[str],
        k: int = 10,
        genre_filter: Optional[List[str]] = None,
        mood_filter: Optional[List[str]] = None,
        fusion_method: str = "weighted_sum",
    ) -> Dict[str, Any]:
        """
        Get hybrid recommendations

        Args:
            track_sequence: User's listening history
            k: Number of recommendations to return
            genre_filter: Optional genre constraints
            mood_filter: Optional mood constraints
            fusion_method: How to combine scores ("weighted_sum", "max", "rank_fusion")

        Returns:
            Dictionary with recommendations and metadata
        """
        if not self._initialized:
            raise ValueError(
                "Hybrid recommender not initialized. Call initialize() first."
            )

        if not track_sequence:
            raise ValueError("track_sequence cannot be empty")

        # Encode session
        session_vector = self.vectorizer.encode_session(track_sequence)

        # Get CF recommendations with larger candidate set
        cf_tracks, cf_scores = self.cf_engine.get_recommendations(
            session_vector,
            k * 3,
            exclude_track_ids=track_sequence,  # Increased from k*2 to k*3
        )

        # Get CB recommendations if available
        cb_tracks, cb_scores = [], []
        if self.cb_engine is not None:
            cb_session_vector = self.cb_engine.encode_session(
                track_sequence, aggregation="mean"
            )
            cb_tracks, cb_scores = self.cb_engine.get_recommendations(
                cb_session_vector,
                k * 3,  # Increased from k*2 to k*3
                exclude_track_ids=track_sequence,
                genre_filter=genre_filter,
                mood_filter=mood_filter,
            )

        # Combine recommendations
        final_recommendations = self._fuse_recommendations(
            cf_tracks, cf_scores, cb_tracks, cb_scores, k, fusion_method
        )

        # Prepare response
        result = {
            "recommendations": [rec["track_id"] for rec in final_recommendations],
            "scores": [rec["score"] for rec in final_recommendations],
            "metadata": {
                "cf_weight": self.cf_weight,
                "cb_weight": self.cb_weight,
                "fusion_method": fusion_method,
                "cf_available": len(cf_tracks) > 0,
                "cb_available": len(cb_tracks) > 0,
                "session_length": len(track_sequence),
            },
        }

        return result

    def _fuse_recommendations(
        self,
        cf_tracks: List[str],
        cf_scores: List[float],
        cb_tracks: List[str],
        cb_scores: List[float],
        k: int,
        method: str,
    ) -> List[Dict[str, Any]]:
        """Fuse CF and CB recommendations using specified method"""

        if method == "weighted_sum":
            return self._weighted_sum_fusion(
                cf_tracks, cf_scores, cb_tracks, cb_scores, k
            )
        elif method == "max":
            return self._max_fusion(cf_tracks, cf_scores, cb_tracks, cb_scores, k)
        elif method == "rank_fusion":
            return self._rank_fusion(cf_tracks, cf_scores, cb_tracks, cb_scores, k)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def _weighted_sum_fusion(
        self,
        cf_tracks: List[str],
        cf_scores: List[float],
        cb_tracks: List[str],
        cb_scores: List[float],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Combine scores using weighted sum"""

        # Normalize scores to [0, 1]
        cf_scores_norm = self._normalize_scores(cf_scores) if cf_scores else []
        cb_scores_norm = self._normalize_scores(cb_scores) if cb_scores else []

        # Create score maps
        cf_score_map = dict(zip(cf_tracks, cf_scores_norm))
        cb_score_map = dict(zip(cb_tracks, cb_scores_norm))

        # Get all unique tracks
        all_tracks = set(cf_tracks + cb_tracks)

        # Calculate hybrid scores
        hybrid_scores = {}
        for track in all_tracks:
            cf_score = cf_score_map.get(track, 0.0)
            cb_score = cb_score_map.get(track, 0.0)

            # Handle case where only one method has the track
            if track in cf_score_map and track not in cb_score_map:
                hybrid_score = self.cf_weight * cf_score
            elif track in cb_score_map and track not in cf_score_map:
                hybrid_score = self.cb_weight * cb_score
            else:
                # Track appears in both - use full weighted combination
                hybrid_score = self.cf_weight * cf_score + self.cb_weight * cb_score
                # Boost tracks that appear in both systems (consensus bonus)
                hybrid_score *= 1.1

            hybrid_scores[track] = hybrid_score

        # Sort and return top-k
        sorted_tracks = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[
            :k
        ]

        return [{"track_id": track, "score": score} for track, score in sorted_tracks]

    def _max_fusion(
        self,
        cf_tracks: List[str],
        cf_scores: List[float],
        cb_tracks: List[str],
        cb_scores: List[float],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Combine scores using max fusion"""

        cf_scores_norm = self._normalize_scores(cf_scores) if cf_scores else []
        cb_scores_norm = self._normalize_scores(cb_scores) if cb_scores else []

        cf_score_map = dict(zip(cf_tracks, cf_scores_norm))
        cb_score_map = dict(zip(cb_tracks, cb_scores_norm))

        all_tracks = set(cf_tracks + cb_tracks)

        hybrid_scores = {}
        for track in all_tracks:
            cf_score = cf_score_map.get(track, 0.0)
            cb_score = cb_score_map.get(track, 0.0)
            hybrid_scores[track] = max(cf_score, cb_score)

        sorted_tracks = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[
            :k
        ]

        return [{"track_id": track, "score": score} for track, score in sorted_tracks]

    def _rank_fusion(
        self,
        cf_tracks: List[str],
        cf_scores: List[float],
        cb_tracks: List[str],
        cb_scores: List[float],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Combine recommendations using rank-based fusion"""

        # Create rank maps (1-indexed)
        cf_rank_map = {track: idx + 1 for idx, track in enumerate(cf_tracks)}
        cb_rank_map = {track: idx + 1 for idx, track in enumerate(cb_tracks)}

        all_tracks = set(cf_tracks + cb_tracks)

        # Calculate reciprocal rank scores
        hybrid_scores = {}
        for track in all_tracks:
            cf_rank = cf_rank_map.get(track, len(cf_tracks) + 1)
            cb_rank = cb_rank_map.get(track, len(cb_tracks) + 1)

            # Reciprocal rank fusion (RRF)
            rrf_score = (1.0 / cf_rank) + (1.0 / cb_rank)
            hybrid_scores[track] = rrf_score

        sorted_tracks = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[
            :k
        ]

        return [{"track_id": track, "score": score} for track, score in sorted_tracks]

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(score - min_score) / (max_score - min_score) for score in scores]

    def get_track_similarities(
        self, track_id: str, k: int = 10, use_cf: bool = True, use_cb: bool = True
    ) -> Dict[str, Any]:
        """Get similar tracks using both CF and CB approaches"""

        if not self._initialized:
            raise ValueError("Hybrid recommender not initialized")

        results = {}

        if use_cf and self.cf_engine:
            try:
                cf_similar, cf_scores = self.cf_engine.get_track_similarities(
                    track_id, k
                )
                results["cf_similar"] = list(zip(cf_similar, cf_scores))
            except Exception as e:
                logger.warning(f"CF track similarity failed: {e}")
                results["cf_similar"] = []

        if use_cb and self.cb_engine:
            try:
                cb_similar, cb_scores = self.cb_engine.get_track_similarities(
                    track_id, k
                )
                results["cb_similar"] = list(zip(cb_similar, cb_scores))
            except Exception as e:
                logger.warning(f"CB track similarity failed: {e}")
                results["cb_similar"] = []

        return results

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""

        if not self._initialized:
            return {"status": "not_initialized"}

        stats = {
            "status": "initialized",
            "weights": {"cf": self.cf_weight, "cb": self.cb_weight},
            "components": {
                "vectorizer": True,
                "cf_engine": self.cf_engine is not None,
                "cb_engine": self.cb_engine is not None,
            },
        }

        if self.cf_engine:
            stats["cf_stats"] = self.cf_engine.get_stats()

        if self.cb_engine:
            stats["cb_stats"] = self.cb_engine.get_stats()

        return stats
