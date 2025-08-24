import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from lightfm.cross_validation import random_train_test_split
from scipy.sparse import csr_matrix
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    loss: str = "warp"  # WARP is better for ranking
    learning_rate: float = 0.1  # Increased from 0.05 for faster convergence
    item_alpha: float = 1e-5  # Reduced regularization for better fit
    user_alpha: float = 1e-5  # Reduced regularization for better fit
    no_components: int = 128  # Keep 128 dimensions
    max_sampled: int = 20  # Increased from 10 for better negative sampling
    epochs: int = 100  # Increased from 50 for better training
    num_threads: int = 4
    random_state: int = 42


class LightFMTrainer:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.model: Optional[LightFM] = None
        self.dataset: Optional[Dataset] = None
        self.user_id_map: Dict[str, int] = {}
        self.item_id_map: Dict[str, int] = {}
        self.reverse_item_map: Dict[int, str] = {}

    def load_session_data(self, session_file: Path) -> List[Dict[str, Any]]:
        logger.info(f"Loading session data from {session_file}")
        with open(session_file) as f:
            if session_file.suffix == ".json":
                sessions = json.load(f)
            elif session_file.suffix == ".jsonl":
                sessions = [json.loads(line) for line in f]
            else:
                raise ValueError(f"Unsupported file format: {session_file.suffix}")
        logger.info(f"Loaded {len(sessions)} sessions")
        return sessions

    def sessions_to_interactions(
        self, sessions: List[Dict[str, Any]], min_interactions: int = 5
    ) -> List[Tuple[str, str, float]]:
        interactions = []
        user_counts = {}
        for session in sessions:
            user_id = session.get("user_id", session.get("session_id", "anonymous"))
            track_sequence = session["track_sequence"]
            user_counts[user_id] = user_counts.get(user_id, 0) + len(track_sequence)
        valid_users = {u for u, c in user_counts.items() if c >= min_interactions}
        logger.info(
            f"Filtered to {len(valid_users)} users with >= {min_interactions} interactions"
        )
        for session in sessions:
            user_id = session.get("user_id", session.get("session_id", "anonymous"))
            if user_id not in valid_users:
                continue
            track_sequence = session["track_sequence"]
            for i, track_id in enumerate(track_sequence):
                # Improved rating scheme: position + frequency + recency
                position_weight = (i + 1) / len(
                    track_sequence
                )  # Later tracks more important
                base_rating = 1.0 + position_weight * 4.0  # Scale 1.0 to 5.0

                # Add frequency bonus for tracks that appear multiple times
                frequency_bonus = track_sequence.count(track_id) * 0.2

                rating = min(base_rating + frequency_bonus, 5.0)  # Cap at 5.0
                interactions.append((user_id, track_id, rating))
        logger.info(f"Generated {len(interactions)} interactions")
        return interactions

    def build_dataset(self, interactions: List[Tuple[str, str, float]]) -> Dataset:
        users = list(set(interaction[0] for interaction in interactions))
        items = list(set(interaction[1] for interaction in interactions))
        logger.info(f"Building dataset with {len(users)} users and {len(items)} items")
        self.dataset = Dataset()
        self.dataset.fit(users=users, items=items)
        self.user_id_map = {uid: i for i, uid in enumerate(users)}
        self.item_id_map = {iid: i for i, iid in enumerate(items)}
        self.reverse_item_map = {i: iid for iid, i in self.item_id_map.items()}
        return self.dataset

    def build_interaction_matrix(
        self, interactions: List[Tuple[str, str, float]]
    ) -> csr_matrix:
        if self.dataset is None:
            raise ValueError("Dataset must be built first")
        interaction_data = []
        missing_users = set()
        missing_items = set()
        for user_id, item_id, rating in interactions:
            if not isinstance(user_id, str):
                logger.error(f"Non-string user_id found: {user_id} ({type(user_id)})")
            if not isinstance(item_id, str):
                logger.error(f"Non-string item_id found: {item_id} ({type(item_id)})")
            if user_id in self.user_id_map and item_id in self.item_id_map:
                interaction_data.append((user_id, item_id, rating))
            else:
                if user_id not in self.user_id_map:
                    missing_users.add(user_id)
                if item_id not in self.item_id_map:
                    missing_items.add(item_id)
        if missing_users:
            logger.error(f"User IDs not in mapping: {sorted(missing_users)}")
        if missing_items:
            logger.error(f"Item IDs not in mapping: {sorted(missing_items)}")
        if missing_users or missing_items:
            raise ValueError(
                "Some user or item IDs in interactions are not in the mapping. See logs for details."
            )
        interactions_matrix, weights = self.dataset.build_interactions(interaction_data)
        logger.info(f"Built interaction matrix: {interactions_matrix.shape}")
        return interactions_matrix, weights

    def train(self, interactions_matrix: csr_matrix) -> LightFM:
        self.model = LightFM(
            loss=self.config.loss,
            learning_rate=self.config.learning_rate,
            item_alpha=self.config.item_alpha,
            user_alpha=self.config.user_alpha,
            no_components=self.config.no_components,
            max_sampled=self.config.max_sampled,
            random_state=self.config.random_state,
        )
        logger.info(f"Training LightFM model for {self.config.epochs} epochs...")
        self.model.fit(
            interactions=interactions_matrix,
            epochs=self.config.epochs,
            num_threads=self.config.num_threads,
            verbose=True,
        )
        logger.info("Training completed!")
        return self.model

    def evaluate(
        self, test_interactions: csr_matrix, train_interactions: csr_matrix, k: int = 10
    ) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model must be trained first")
        logger.info(f"Evaluating model performance at k={k}")
        precision = precision_at_k(
            self.model, test_interactions, train_interactions, k=k
        ).mean()
        recall = recall_at_k(
            self.model, test_interactions, train_interactions, k=k
        ).mean()
        auc = auc_score(self.model, test_interactions, train_interactions).mean()
        metrics = {
            f"precision_at_{k}": float(precision),
            f"recall_at_{k}": float(recall),
            "auc": float(auc),
        }
        logger.info(f"Evaluation results: {metrics}")
        return metrics

    def extract_item_embeddings(self) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise ValueError("Model must be trained first")
        item_embeddings = self.model.item_embeddings
        embeddings_dict = {
            item_id: item_embeddings[item_idx]
            for item_id, item_idx in self.item_id_map.items()
        }
        logger.info(f"Extracted embeddings for {len(embeddings_dict)} items")
        return embeddings_dict

    def save_model(self, model_path: Path) -> None:
        if self.model is None:
            raise ValueError("No model to save")
        model_data = {
            "model": self.model,
            "config": self.config,
            "user_id_map": self.user_id_map,
            "item_id_map": self.item_id_map,
            "reverse_item_map": self.reverse_item_map,
            "dataset": self.dataset,
        }
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {model_path}")

    def save_embeddings(self, output_path: Path, format: str = "json") -> None:
        embeddings = self.extract_item_embeddings()
        if format == "json":
            embeddings_serializable = {
                item_id: embedding.tolist() for item_id, embedding in embeddings.items()
            }
            with open(output_path, "w") as f:
                json.dump(embeddings_serializable, f, indent=2)
        elif format == "npy":
            np.savez(output_path, **embeddings)
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.info(f"Embeddings saved to {output_path} in {format} format")


def train_lightfm_model(
    session_file: Path,
    output_dir: Path,
    config: Optional[TrainingConfig] = None,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = LightFMTrainer(config)
    sessions = trainer.load_session_data(session_file)
    interactions = trainer.sessions_to_interactions(sessions)
    interactions = [(str(u), str(i), r) for (u, i, r) in interactions]
    dataset = trainer.build_dataset(interactions)
    interactions_matrix, weights = trainer.build_interaction_matrix(interactions)
    train_interactions, test_interactions = random_train_test_split(
        interactions_matrix, test_percentage=test_size, random_state=42
    )
    model = trainer.train(train_interactions)
    model_path = output_dir / "lightfm_model.pkl"
    embeddings_path = output_dir / "lightfm_embeddings.json"
    trainer.save_model(model_path)
    trainer.save_embeddings(embeddings_path, format="json")
    logger.info(f"Training complete! Model saved to {output_dir}")
    return {
        "model_path": model_path,
        "embeddings_path": embeddings_path,
        "num_users": len(trainer.user_id_map),
        "num_items": len(trainer.item_id_map),
        "num_interactions": len(interactions),
    }
