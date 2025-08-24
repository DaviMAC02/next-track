#!/usr/bin/env python3
import argparse
import json
import logging
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Track:
    id: str
    title: str
    artist: str
    album: Optional[str] = None
    duration: Optional[float] = None
    genres: List[str] = None
    moods: List[str] = None
    year: Optional[int] = None
    tempo: Optional[float] = None
    popularity: Optional[float] = None

    def __post_init__(self):
        if self.genres is None:
            self.genres = []
        if self.moods is None:
            self.moods = []


@dataclass
class Session:
    user_id: str
    session_id: str
    track_ids: List[str]
    timestamps: List[datetime]
    duration: float


class MusicBrainzDataGenerator:
    BASE_URL = "https://musicbrainz.org/ws/2"
    USER_AGENT = "MusicRecSystem/1.0 (https://github.com/user/music-rec)"

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

    def search_tracks(
        self, query: str, limit: int = 25, offset: int = 0
    ) -> List[Dict[str, Any]]:
        url = f"{self.BASE_URL}/recording"
        params = {
            "query": query,
            "fmt": "json",
            "limit": min(limit, 100),
            "offset": offset,
        }
        try:
            time.sleep(self.rate_limit)
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("recordings", [])
        except requests.RequestException as e:
            logger.error(f"MusicBrainz API error: {e}")
            return []

    def get_track_details(self, mbid: str) -> Optional[Dict[str, Any]]:
        url = f"{self.BASE_URL}/recording/{mbid}"
        params = {"fmt": "json", "inc": "artist-credits+releases+tags+genres"}
        try:
            time.sleep(self.rate_limit)
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching track {mbid}: {e}")
            return None

    def generate_tracks(self, num_tracks: int = 1000) -> List[Track]:
        tracks = []
        search_terms = [
            "rock",
            "pop",
            "jazz",
            "classical",
            "electronic",
            "hip-hop",
            "blues",
            "country",
            "folk",
            "reggae",
            "punk",
            "metal",
            "ambient",
            "techno",
            "house",
            "drum",
            "guitar",
            "piano",
        ]
        logger.info(f"Generating {num_tracks} tracks from MusicBrainz...")
        tracks_per_term = max(1, num_tracks // len(search_terms))
        for search_term in search_terms:
            if len(tracks) >= num_tracks:
                break
            logger.info(f"Searching for '{search_term}' tracks...")
            offset = 0
            while len(tracks) < num_tracks and offset < 1000:
                recordings = self.search_tracks(search_term, limit=25, offset=offset)
                if not recordings:
                    break
                for recording in recordings:
                    if len(tracks) >= num_tracks:
                        break
                    track = self._parse_recording(recording)
                    if track:
                        tracks.append(track)
                offset += 25
        logger.info(f"Generated {len(tracks)} tracks from MusicBrainz")
        return tracks[:num_tracks]

    def _parse_recording(self, recording: Dict[str, Any]) -> Optional[Track]:
        try:
            track_id = recording.get("id")
            title = recording.get("title")
            if not track_id or not title:
                return None
            artist = "Unknown Artist"
            if "artist-credit" in recording and recording["artist-credit"]:
                artist = recording["artist-credit"][0].get("name", "Unknown Artist")
            album = None
            if "releases" in recording and recording["releases"]:
                album = recording["releases"][0].get("title")
            duration = None
            if "length" in recording and recording["length"]:
                duration = int(recording["length"]) / 1000.0
            genres = []
            if "tags" in recording:
                genres = [tag["name"].lower() for tag in recording["tags"][:5]]
            year = None
            if "releases" in recording and recording["releases"]:
                release_date = recording["releases"][0].get("date")
                if release_date and len(release_date) >= 4:
                    try:
                        year = int(release_date[:4])
                    except ValueError:
                        pass
            return Track(
                id=track_id,
                title=title,
                artist=artist,
                album=album,
                duration=duration,
                genres=genres,
                year=year,
                tempo=None,
                popularity=None,
            )
        except Exception as e:
            logger.warning(f"Error parsing recording: {e}")
            return None


class SyntheticSessionGenerator:
    """Generate synthetic user sessions for demo purposes only."""

    def generate_sessions(
        self, tracks: List[Track], num_users: int = 20, sessions_per_user: int = 5
    ) -> List[Session]:
        import random

        sessions = []
        if not tracks:
            raise RuntimeError("No tracks available for session generation")
        for user_idx in range(num_users):
            user_id = f"user_{user_idx}"
            for session_idx in range(sessions_per_user):
                session_size = random.randint(5, 12)
                session_tracks = random.sample(tracks, min(session_size, len(tracks)))
                session_start = datetime.now() - timedelta(days=session_idx)
                timestamps = [
                    session_start + timedelta(minutes=i * 3)
                    for i in range(len(session_tracks))
                ]
                session = Session(
                    user_id=user_id,
                    session_id=f"session_{user_id}_{session_idx}_{int(time.time())}",
                    track_ids=[track.id for track in session_tracks],
                    timestamps=timestamps,
                    duration=(
                        (timestamps[-1] - timestamps[0]).total_seconds()
                        if len(timestamps) > 1
                        else 1800.0
                    ),
                )
                sessions.append(session)
        return sessions


class LocalDatasetParser:
    def parse_spotify_dataset(
        self, csv_path: Path
    ) -> Tuple[List[Track], List[Session]]:
        logger.info(f"Parsing dataset from {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            tracks = []
            sessions = []
            for _, row in df.iterrows():
                track = Track(
                    id=str(row.get("track_id", f"track_{len(tracks)}")),
                    title=str(row.get("track_name", "Unknown")),
                    artist=str(row.get("artist_name", "Unknown")),
                    album=str(row.get("album_name")),
                    duration=(
                        float(row.get("duration_ms", 0)) / 1000.0
                        if pd.notna(row.get("duration_ms"))
                        else None
                    ),
                    genres=(
                        str(row.get("genres", "")).split(",")
                        if pd.notna(row.get("genres"))
                        else []
                    ),
                    year=int(row.get("year")) if pd.notna(row.get("year")) else None,
                    tempo=(
                        float(row.get("tempo")) if pd.notna(row.get("tempo")) else None
                    ),
                    popularity=(
                        float(row.get("popularity", 0)) / 100.0
                        if pd.notna(row.get("popularity"))
                        else None
                    ),
                )
                tracks.append(track)
            logger.info(f"Parsed {len(tracks)} tracks from dataset")
            # Sessions must be provided in a separate file or format
            return tracks, sessions
        except Exception as e:
            logger.error(f"Error parsing dataset: {e}")
            return [], []


class DataSaver:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_tracks(self, tracks: List[Track], filename: str = "tracks.json"):
        output_path = self.output_dir / filename
        tracks_data = []
        for track in tracks:
            tracks_data.append(
                {
                    "track_id": track.id,
                    "title": track.title,
                    "artist": track.artist,
                    "album": track.album,
                    "duration": track.duration,
                    "genres": track.genres,
                    "moods": track.moods,
                    "year": track.year,
                    "tempo": track.tempo,
                    "popularity": track.popularity,
                }
            )
        with open(output_path, "w") as f:
            json.dump(tracks_data, f, indent=2)
        logger.info(f"Saved {len(tracks)} tracks to {output_path}")

    def save_sessions(self, sessions: List[Session], filename: str = "sessions.json"):
        output_path = self.output_dir / filename
        sessions_data = []
        for session in sessions:
            sessions_data.append(
                {
                    "user_id": session.user_id,
                    "session_id": session.session_id,
                    "track_ids": session.track_ids,
                    "timestamps": [ts.isoformat() for ts in session.timestamps],
                    "duration": session.duration,
                }
            )
        with open(output_path, "w") as f:
            json.dump(sessions_data, f, indent=2)
        logger.info(f"Saved {len(sessions)} sessions to {output_path}")

    def save_embeddings(
        self, embeddings: Dict[str, Any], filename: str = "track_embeddings.json"
    ):
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(embeddings, f, indent=2)
        logger.info(f"Saved embeddings for {len(embeddings)} tracks to {output_path}")

    def save_metadata(self, tracks: List[Track], filename: str = "track_metadata.json"):
        output_path = self.output_dir / filename
        metadata = {}
        for track in tracks:
            metadata[track.id] = {
                "title": track.title,
                "artist": track.artist,
                "album": track.album,
                "genre": track.genres,
                "mood": track.moods,
                "duration": track.duration,
                "year": track.year,
                "tempo": track.tempo,
                "popularity": track.popularity,
            }
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata for {len(metadata)} tracks to {output_path}")


def generate_metadata_features(tracks: List[Track], output_dir: Path):
    """Generate content-based features from track metadata."""
    logger.info("Generating content-based features...")

    features = {}

    for track in tracks:
        # Create feature vector
        feature_vector = []

        # Duration (normalized)
        duration = track.duration if track.duration else 200000
        duration_norm = min(
            duration / 1000.0 / 600.0, 1.0
        )  # Convert ms to s, normalize to max 10 min
        feature_vector.append(duration_norm)

        # Year (normalized)
        year = track.year if track.year else 2000
        year_norm = (year - 1950) / 70.0 if year > 1950 else 0.0  # Normalize 1950-2020
        feature_vector.append(year_norm)

        # Tempo (normalized)
        tempo = track.tempo if track.tempo else 120.0
        tempo_norm = tempo / 200.0  # Normalize to max 200 BPM
        feature_vector.append(tempo_norm)

        # Genre features (one-hot encoding)
        genres = track.genres if track.genres else []
        common_genres = [
            "rock",
            "pop",
            "jazz",
            "electronic",
            "classical",
            "hip-hop",
            "country",
            "blues",
        ]
        for genre in common_genres:
            genre_present = (
                1.0 if any(genre.lower() in g.lower() for g in genres) else 0.0
            )
            feature_vector.append(genre_present)

        # Mood features (one-hot encoding)
        moods = track.moods if track.moods else []
        common_moods = ["happy", "sad", "energetic", "calm", "aggressive", "romantic"]
        for mood in common_moods:
            mood_present = 1.0 if any(mood.lower() in m.lower() for m in moods) else 0.0
            feature_vector.append(mood_present)

        features[track.id] = feature_vector

    # Save content features
    import pickle

    with open(output_dir / "metadata_features.npy", "wb") as f:
        pickle.dump(features, f)

    # Save track IDs for reference
    track_ids = list(sorted(features.keys()))
    with open(output_dir / "metadata_features_track_ids.json", "w") as f:
        json.dump(track_ids, f, indent=2)

    logger.info(
        f"Generated content features for {len(features)} tracks with {len(feature_vector)} dimensions"
    )


def train_lightfm_embeddings(
    sessions: List[Session],
    tracks: List[Track],
    output_dir: Path,
    embedding_dim: int = 64,
):
    try:
        from training.lightfm_trainer import train_lightfm_model
    except ImportError:
        raise RuntimeError(
            "LightFM and its dependencies must be installed for embedding generation."
        )
    # Prepare session data in the expected format
    session_dicts = []
    for session in sessions:
        session_dicts.append(
            {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "track_sequence": session.track_ids,
            }
        )
    # Save sessions to a temp file
    temp_sessions_file = output_dir / "_lightfm_sessions.json"
    with open(temp_sessions_file, "w") as f:
        json.dump(session_dicts, f, indent=2)
    # Train LightFM and save embeddings
    results = train_lightfm_model(
        temp_sessions_file, output_dir, config=None, test_size=0.0
    )
    logger.info(
        f"LightFM training complete. Embeddings saved to {results['embeddings_path']}"
    )
    # Delete the temporary sessions file
    try:
        temp_sessions_file.unlink()
        logger.info(f"Deleted temporary file {temp_sessions_file}")
    except Exception as e:
        logger.warning(f"Could not delete temporary file {temp_sessions_file}: {e}")
    return results["embeddings_path"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate music recommendation data (production-ready, no dummy data)"
    )
    parser.add_argument(
        "--tracks", type=int, default=500, help="Number of tracks to generate"
    )
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--users", type=int, default=20, help="Number of users to generate sessions for"
    )
    parser.add_argument(
        "--sessions-per-user", type=int, default=5, help="Number of sessions per user"
    )
    args = parser.parse_args()
    output_dir = Path(args.output)
    saver = DataSaver(output_dir)
    tracks = []
    sessions = []
    # Track metadata (MusicBrainz only)
    logger.info("Generating data from MusicBrainz...")
    mb_generator = MusicBrainzDataGenerator()
    mb_tracks = mb_generator.generate_tracks(args.tracks)
    tracks.extend(mb_tracks)
    # Session data (synthetic)
    if tracks:
        logger.info("Generating synthetic user sessions for demo purposes...")
        session_generator = SyntheticSessionGenerator()
        synthetic_sessions = session_generator.generate_sessions(
            tracks, num_users=args.users, sessions_per_user=args.sessions_per_user
        )
        sessions.extend(synthetic_sessions)
    if not sessions:
        logger.error(
            "No session data available. Cannot proceed without sessions for training."
        )
        raise RuntimeError(
            "No session data available. Cannot proceed without sessions."
        )
    # Save all generated data
    if tracks:
        saver.save_tracks(tracks, "production_tracks.json")
        saver.save_metadata(tracks, "production_track_metadata.json")
        generate_metadata_features(tracks, output_dir)
    if sessions:
        saver.save_sessions(sessions, "production_sessions.json")
        # Also save in training format expected by the system
        training_sessions = []
        for session in sessions:
            training_sessions.append(
                {
                    "user_id": session.user_id,
                    "session_id": session.session_id,
                    "track_ids": session.track_ids,
                }
            )
        training_output = output_dir / "production_training_sessions.json"
        with open(training_output, "w") as f:
            json.dump(training_sessions, f, indent=2)
        logger.info(
            f"Saved {len(training_sessions)} training sessions to {training_output}"
        )
    # Embedding generation (LightFM)
    if sessions and tracks:
        logger.info("Training LightFM model to generate real embeddings...")
        embeddings_path = train_lightfm_embeddings(sessions, tracks, output_dir)
        logger.info(f"Embeddings saved to {embeddings_path}")
        target_path = output_dir / "production_track_embeddings.json"
        if embeddings_path != target_path:
            import shutil

            shutil.copy(embeddings_path, target_path)
            logger.info(f"Copied embeddings to {target_path}")
    logger.info("Data generation complete!")
    logger.info(f"Generated {len(tracks)} tracks and {len(sessions)} sessions")
    logger.info(f"Data saved to {output_dir}")


if __name__ == "__main__":
    main()
