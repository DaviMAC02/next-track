import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ContentBasedEngine:
    """Content-based recommendation engine using audio or metadata features"""
    
    def __init__(self, feature_type: str = "metadata"):
        """
        Initialize content-based engine
        
        Args:
            feature_type: "audio", "metadata", or "combined"
        """
        self.feature_type = feature_type
        self.track_ids: List[str] = []
        self.features: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        self._track_to_idx: Dict[str, int] = {}
        self.genre_encoder: Optional[Dict[str, int]] = None
        self.mood_encoder: Optional[Dict[str, int]] = None
        self.metadata: Optional[List[Dict[str, Any]]] = None
    
    def load_features(self, features_path: Path, track_ids: List[str] = None) -> None:
        """Load content features from file"""
        try:
            if features_path.suffix == '.npy':
                # NumPy format (pickled dict)
                features_dict = np.load(features_path, allow_pickle=True)
                self.track_ids = list(features_dict.keys())
                features = np.vstack([features_dict[tid] for tid in self.track_ids])
            elif features_path.suffix == '.json':
                # JSON format with track_id -> features mapping
                with open(features_path) as f:
                    features_dict = json.load(f)
                self.track_ids = list(features_dict.keys())
                features = np.vstack([features_dict[tid] for tid in self.track_ids])
            else:
                raise ValueError(f"Unsupported file format: {features_path.suffix}")
            
            # Store features and create index mapping
            self.features = features
            self._track_to_idx = {tid: idx for idx, tid in enumerate(self.track_ids)}
            
            logger.info(f"Loaded {self.feature_type} features for {len(self.track_ids)} tracks "
                       f"({features.shape[1]} dimensions) from {features_path}")
            
        except Exception as e:
            logger.error(f"Failed to load content features: {e}")
            raise
    
    def load_metadata(self, metadata_path: Path) -> None:
        """Load track metadata for genre/mood filtering"""
        try:
            with open(metadata_path) as f:
                metadata_dict = json.load(f)
            
            # Convert to list format aligned with track_ids
            self.metadata = []
            for track_id in self.track_ids:
                if track_id in metadata_dict:
                    self.metadata.append(metadata_dict[track_id])
                else:
                    # Default metadata for missing tracks
                    self.metadata.append({'genre': [], 'mood': []})
            
            # Build encoders for genre and mood
            all_genres = set()
            all_moods = set()
            
            for meta in self.metadata:
                if 'genre' in meta:
                    genres = meta['genre'] if isinstance(meta['genre'], list) else [meta['genre']]
                    all_genres.update(genres)
                if 'mood' in meta:
                    moods = meta['mood'] if isinstance(meta['mood'], list) else [meta['mood']]
                    all_moods.update(moods)
            
            self.genre_encoder = {genre: idx for idx, genre in enumerate(sorted(all_genres))}
            self.mood_encoder = {mood: idx for idx, mood in enumerate(sorted(all_moods))}
            
            logger.info(f"Loaded metadata for {len(self.metadata)} tracks with "
                       f"{len(all_genres)} genres and {len(all_moods)} moods")
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
    
    def build_index(self, normalize_features: bool = True) -> None:
        """Build content-based index (preprocessing)"""
        if self.features is None:
            raise ValueError("Features not loaded. Call load_features first.")
        
        if normalize_features:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
            logger.info("Normalized content features")
        
        logger.info(f"Content-based engine built with {len(self.track_ids)} tracks")
    
    def get_track_vector(self, track_id: str) -> np.ndarray:
        """Get content vector for a specific track"""
        if track_id not in self._track_to_idx:
            raise ValueError(f"Unknown track ID: {track_id}")
        
        idx = self._track_to_idx[track_id]
        return self.features[idx:idx+1]
    
    def encode_session(self, track_ids: List[str], aggregation: str = "mean") -> np.ndarray:
        """
        Encode session based on content features
        
        Args:
            track_ids: List of track IDs in the session
            aggregation: "mean", "max", "last" - how to aggregate track features
        """
        if self.features is None:
            raise ValueError("Features not loaded and indexed")
        
        if not track_ids:
            raise ValueError("track_ids cannot be empty")
        
        # Get feature vectors for session tracks
        session_features = []
        for track_id in track_ids:
            if track_id in self._track_to_idx:
                idx = self._track_to_idx[track_id]
                session_features.append(self.features[idx])
            else:
                logger.warning(f"Track {track_id} not found in content features, skipping")
        
        if not session_features:
            raise ValueError("No valid tracks found in content features")
        
        session_matrix = np.array(session_features)
        
        # Aggregate session features
        if aggregation == "mean":
            session_vector = session_matrix.mean(axis=0)
        elif aggregation == "max":
            session_vector = session_matrix.max(axis=0)
        elif aggregation == "last":
            session_vector = session_matrix[-1]  # Last track
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        return session_vector.reshape(1, -1)
    
    def get_recommendations(self, session_vector: np.ndarray, k: int, 
                          exclude_track_ids: List[str] = None,
                          genre_filter: List[str] = None,
                          mood_filter: List[str] = None) -> Tuple[List[str], List[float]]:
        """
        Get content-based recommendations
        
        Args:
            session_vector: Session representation vector
            k: Number of recommendations
            exclude_track_ids: Track IDs to exclude from recommendations
            genre_filter: Optional genre filter (placeholder)
            mood_filter: Optional mood filter (placeholder)
        """
        if self.features is None:
            raise ValueError("Features not loaded and indexed")
        
        # Compute content similarities
        similarities = cosine_similarity(session_vector, self.features).flatten()
        
        # Apply exclusions
        exclude_indices = set()
        if exclude_track_ids:
            for track_id in exclude_track_ids:
                if track_id in self._track_to_idx:
                    exclude_indices.add(self._track_to_idx[track_id])
        
        for idx in exclude_indices:
            similarities[idx] = -1.0
        
        # Apply genre/mood filters when metadata is available
        if (genre_filter or mood_filter) and self.metadata:
            filtered_indices = self._filter_by_genre_mood(genre_filter, mood_filter)
            for idx in range(len(self.track_ids)):
                if idx not in filtered_indices:
                    similarities[idx] = -1.0
        
        # Get top-k
        top_indices = similarities.argsort()[::-1][:k]
        top_scores = similarities[top_indices]
        
        recommended_tracks = [self.track_ids[idx] for idx in top_indices]
        
        return recommended_tracks, top_scores.tolist()
    
    def _filter_by_genre_mood(self, genre_filter: List[str] = None, mood_filter: List[str] = None) -> set:
        """Filter track indices by genre and mood criteria"""
        valid_indices = set()
        
        for idx, meta in enumerate(self.metadata):
            include_track = True
            
            # Check genre filter
            if genre_filter:
                track_genres = meta.get('genre', [])
                if isinstance(track_genres, str):
                    track_genres = [track_genres]
                
                if not any(genre in track_genres for genre in genre_filter):
                    include_track = False
            
            # Check mood filter
            if mood_filter and include_track:
                track_moods = meta.get('mood', [])
                if isinstance(track_moods, str):
                    track_moods = [track_moods]
                
                if not any(mood in track_moods for mood in mood_filter):
                    include_track = False
            
            if include_track:
                valid_indices.add(idx)
        
        return valid_indices
    
    def get_track_similarities(self, track_id: str, k: int = 10) -> Tuple[List[str], List[float]]:
        """Get tracks similar to a specific track based on content"""
        track_vector = self.get_track_vector(track_id)
        return self.get_recommendations(track_vector, k, exclude_track_ids=[track_id])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "feature_type": self.feature_type,
            "num_tracks": len(self.track_ids),
            "feature_dimensions": self.features.shape[1] if self.features is not None else 0,
            "normalized": self.scaler is not None,
            "track_ids_sample": self.track_ids[:5] if self.track_ids else []
        }

class MetadataFeaturesExtractor:
    """Extract features from track metadata"""
    
    def __init__(self):
        self.genre_encoder: Optional[Dict[str, int]] = None
        self.artist_encoder: Optional[Dict[str, int]] = None
    
    def extract_features(self, metadata: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract features from track metadata (placeholder implementation)
        
        Args:
            metadata: List of metadata dictionaries for each track
            
        Returns:
            Feature matrix (n_tracks, n_features)
        """
        # Placeholder implementation
        features = []
        
        # Build encoders
        all_genres = set()
        all_artists = set()
        
        for meta in metadata:
            if 'genre' in meta:
                all_genres.update(meta['genre'] if isinstance(meta['genre'], list) else [meta['genre']])
            if 'artist' in meta:
                all_artists.add(meta['artist'])
        
        self.genre_encoder = {genre: idx for idx, genre in enumerate(sorted(all_genres))}
        self.artist_encoder = {artist: idx for idx, artist in enumerate(sorted(all_artists))}
        
        for meta in metadata:
            # Genre one-hot encoding
            genre_features = np.zeros(len(self.genre_encoder))
            if 'genre' in meta:
                genres = meta['genre'] if isinstance(meta['genre'], list) else [meta['genre']]
                for genre in genres:
                    if genre in self.genre_encoder:
                        genre_features[self.genre_encoder[genre]] = 1.0
            
            # Artist one-hot encoding
            artist_features = np.zeros(len(self.artist_encoder))
            if 'artist' in meta and meta['artist'] in self.artist_encoder:
                artist_features[self.artist_encoder[meta['artist']]] = 1.0
            
            # Numerical features
            duration = meta.get('duration', 0.0) / 300.0  # Normalize by 5 minutes
            year = (meta.get('year', 2000) - 1900) / 100.0  # Normalize year
            tempo = meta.get('tempo', 120.0) / 200.0  # Normalize tempo
            
            # Combine all features
            track_features = np.concatenate([
                genre_features,
                artist_features,
                [duration, year, tempo]
            ])
            
            features.append(track_features)
        
        return np.array(features)
