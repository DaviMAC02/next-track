"""
Offline evaluation harness for NextTrack recommendation system.
Computes standard recommendation metrics like precision@K, recall@K, NDCG@K.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import time
from collections import defaultdict
import random

from api.hybrid_recommender import HybridRecommender
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""

    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    coverage: float
    diversity: float
    novelty: float
    total_users: int
    total_items: int
    total_interactions: int
    evaluation_time: float


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""

    k_values: List[int] = None
    train_ratio: float = 0.8
    min_interactions: int = 5
    max_users: int = None  # Limit for faster evaluation
    random_seed: int = 42

    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10, 20]


class OfflineEvaluator:
    """Offline evaluation system for recommendation models"""

    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.recommender: Optional[HybridRecommender] = None

    def load_sessions(self, sessions_path: Path) -> List[Dict[str, Any]]:
        """Load session data for evaluation"""
        logger.info(f"Loading sessions from {sessions_path}")

        with open(sessions_path) as f:
            if sessions_path.suffix == ".json":
                sessions = json.load(f)
            elif sessions_path.suffix == ".jsonl":
                sessions = [json.loads(line) for line in f]
            else:
                raise ValueError(f"Unsupported format: {sessions_path.suffix}")

        logger.info(f"Loaded {len(sessions)} sessions")
        return sessions

    def preprocess_sessions(
        self, sessions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter and preprocess sessions for evaluation"""
        valid_sessions = []
        for session in sessions:
            track_sequence = session.get("track_sequence", session.get("track_ids", []))
            if len(track_sequence) >= self.config.min_interactions:
                normalized_session = session.copy()
                normalized_session["track_sequence"] = track_sequence
                valid_sessions.append(normalized_session)

        logger.info(
            f"Filtered to {len(valid_sessions)} sessions with >= {self.config.min_interactions} tracks"
        )

        if self.config.max_users:
            user_sessions = defaultdict(list)
            for session in valid_sessions:
                user_id = session.get("user_id", "unknown")
                user_sessions[user_id].append(session)

            if len(user_sessions) > self.config.max_users:
                random.seed(self.config.random_seed)
                selected_users = random.sample(
                    list(user_sessions.keys()), self.config.max_users
                )
                valid_sessions = []
                for user_id in selected_users:
                    valid_sessions.extend(user_sessions[user_id])
                logger.info(
                    f"Sampled {len(selected_users)} users with {len(valid_sessions)} sessions for evaluation"
                )
            else:
                logger.info(
                    f"Using all {len(user_sessions)} users with {len(valid_sessions)} sessions"
                )

        return valid_sessions

    def train_test_split(
        self, sessions: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split sessions into train and test sets using session-level split (not per-session)"""
        # Use session-level split like FINAL_REPORT.md: 80% sessions for train, 20% for test
        random.seed(self.config.random_seed)

        # Shuffle sessions to ensure random split
        shuffled_sessions = sessions.copy()
        random.shuffle(shuffled_sessions)

        # Split at session level
        train_size = int(len(shuffled_sessions) * self.config.train_ratio)
        train_sessions = shuffled_sessions[:train_size]

        # For test sessions, use the entire session as input and last 1-2 tracks as ground truth
        test_sessions = []
        for session in shuffled_sessions[train_size:]:
            track_sequence = session["track_sequence"]

            # Need at least 3 tracks to create meaningful test
            if len(track_sequence) < 3:
                continue

            # Use all but last 1-2 tracks as input, last tracks as ground truth
            if len(track_sequence) <= 5:
                # For short sessions, use last 1 track as ground truth
                input_tracks = track_sequence[:-1]
                ground_truth = track_sequence[-1:]
            else:
                # For longer sessions, use last 2 tracks as ground truth
                input_tracks = track_sequence[:-2]
                ground_truth = track_sequence[-2:]

            test_session = session.copy()
            test_session["track_sequence"] = input_tracks
            test_session["ground_truth"] = ground_truth
            test_sessions.append(test_session)

        logger.info(
            f"Session-level split: {len(train_sessions)} train sessions, {len(test_sessions)} test sessions"
        )
        return train_sessions, test_sessions

    def initialize_recommender(self) -> None:
        """Initialize the hybrid recommender"""
        logger.info("Initializing hybrid recommender for evaluation")

        self.recommender = HybridRecommender(
            cf_weight=settings.CF_WEIGHT, cb_weight=settings.CB_WEIGHT
        )

        # Initialize with existing embeddings and features
        content_features_path = None
        if settings.USE_CONTENT_BASED and settings.METADATA_FEATURES_FILE.exists():
            content_features_path = settings.METADATA_FEATURES_FILE

        index_type = (
            "faiss"
            if settings.USE_ANN_INDEX and settings.ANN_INDEX_TYPE == "faiss"
            else "brute_force"
        )

        self.recommender.initialize(
            embeddings_path=settings.EMBEDDINGS_FILE,
            cf_index_type=index_type,
            use_content_based=settings.USE_CONTENT_BASED,
            content_features_path=content_features_path,
        )

    def compute_precision_recall(
        self, recommendations: List[str], ground_truth: List[str], k: int
    ) -> Tuple[float, float]:
        """Compute precision@k and recall@k"""
        rec_k = recommendations[:k]

        # Intersection
        hits = len(set(rec_k) & set(ground_truth))

        # Precision@k
        precision = hits / k if k > 0 else 0.0

        # Recall@k
        recall = hits / len(ground_truth) if len(ground_truth) > 0 else 0.0

        return precision, recall

    def compute_ndcg(
        self, recommendations: List[str], ground_truth: List[str], k: int
    ) -> float:
        """Compute NDCG@k (Normalized Discounted Cumulative Gain)"""
        rec_k = recommendations[:k]

        # Compute DCG
        dcg = 0.0
        for i, rec in enumerate(rec_k):
            if rec in ground_truth:
                dcg += 1.0 / np.log2(i + 2)

        # Compute IDCG (perfect ranking)
        idcg = 0.0
        for i in range(min(k, len(ground_truth))):
            idcg += 1.0 / np.log2(i + 2)

        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg

    def compute_coverage(
        self, all_recommendations: List[List[str]], catalog_size: int
    ) -> float:
        """Compute catalog coverage"""
        unique_recommended = set()
        for recs in all_recommendations:
            unique_recommended.update(recs)

        coverage = len(unique_recommended) / catalog_size if catalog_size > 0 else 0.0
        return coverage

    def compute_diversity(self, recommendations: List[List[str]]) -> float:
        """Compute intra-list diversity (average pairwise dissimilarity)"""
        if not self.recommender or not self.recommender.vectorizer:
            return 0.0

        diversities = []
        embeddings_matrix = self.recommender.vectorizer.get_embeddings_matrix()
        track_ids = self.recommender.vectorizer.get_track_ids()
        track_to_idx = {tid: idx for idx, tid in enumerate(track_ids)}

        for rec_list in recommendations:
            if len(rec_list) < 2:
                continue

            # Get embeddings for recommended tracks
            valid_recs = [r for r in rec_list if r in track_to_idx]
            if len(valid_recs) < 2:
                continue

            indices = [track_to_idx[r] for r in valid_recs]
            rec_embeddings = embeddings_matrix[indices]

            # Compute pairwise similarities
            similarities = []
            for i in range(len(rec_embeddings)):
                for j in range(i + 1, len(rec_embeddings)):
                    sim = np.dot(rec_embeddings[i], rec_embeddings[j])
                    similarities.append(sim)

            # Diversity = 1 - average similarity
            avg_similarity = np.mean(similarities) if similarities else 0.0
            diversity = 1.0 - avg_similarity
            diversities.append(diversity)

        return np.mean(diversities) if diversities else 0.0

    def compute_novelty(
        self, recommendations: List[List[str]], item_popularity: Dict[str, int]
    ) -> float:
        """Compute novelty (inverse popularity)"""
        total_items = sum(item_popularity.values())

        novelties = []
        for rec_list in recommendations:
            rec_novelties = []
            for item in rec_list:
                if item in item_popularity:
                    popularity = item_popularity[item] / total_items
                    novelty = -np.log2(popularity) if popularity > 0 else 0.0
                    rec_novelties.append(novelty)

            if rec_novelties:
                novelties.append(np.mean(rec_novelties))

        return np.mean(novelties) if novelties else 0.0

    def evaluate(
        self, sessions_path: Path, output_path: Optional[Path] = None
    ) -> EvaluationMetrics:
        """Run complete offline evaluation"""
        start_time = time.time()

        sessions = self.load_sessions(sessions_path)
        sessions = self.preprocess_sessions(sessions)
        train_sessions, test_sessions = self.train_test_split(sessions)

        # Initialize recommender
        self.initialize_recommender()

        # Compute item popularity for novelty
        item_counts = defaultdict(int)
        for session in sessions:
            for track in session["track_sequence"]:
                item_counts[track] += 1

        all_recommendations = []
        precision_scores = {k: [] for k in self.config.k_values}
        recall_scores = {k: [] for k in self.config.k_values}
        ndcg_scores = {k: [] for k in self.config.k_values}

        logger.info(f"Evaluating on {len(test_sessions)} test sessions")

        for i, test_session in enumerate(test_sessions):
            if i % 100 == 0:
                logger.info(f"Evaluated {i}/{len(test_sessions)} sessions")

            try:
                # Get recommendations
                max_k = max(self.config.k_values)
                result = self.recommender.get_recommendations(
                    track_sequence=test_session["track_sequence"],
                    k=max_k,
                    fusion_method="weighted_sum",
                )

                recommendations = result["recommendations"]
                ground_truth = test_session["ground_truth"]

                all_recommendations.append(recommendations)

                # Compute metrics for each k
                for k in self.config.k_values:
                    precision, recall = self.compute_precision_recall(
                        recommendations, ground_truth, k
                    )
                    ndcg = self.compute_ndcg(recommendations, ground_truth, k)

                    precision_scores[k].append(precision)
                    recall_scores[k].append(recall)
                    ndcg_scores[k].append(ndcg)

            except Exception as e:
                logger.warning(f"Failed to evaluate session {i}: {e}")
                continue

        # Aggregate metrics
        catalog_size = len(
            set(track for session in sessions for track in session["track_sequence"])
        )

        metrics = EvaluationMetrics(
            precision_at_k={
                k: np.mean(scores) for k, scores in precision_scores.items()
            },
            recall_at_k={k: np.mean(scores) for k, scores in recall_scores.items()},
            ndcg_at_k={k: np.mean(scores) for k, scores in ndcg_scores.items()},
            coverage=self.compute_coverage(all_recommendations, catalog_size),
            diversity=self.compute_diversity(all_recommendations),
            novelty=self.compute_novelty(all_recommendations, item_counts),
            total_users=len(sessions),
            total_items=catalog_size,
            total_interactions=sum(len(s["track_sequence"]) for s in sessions),
            evaluation_time=time.time() - start_time,
        )

        # Save results
        if output_path:
            self.save_results(metrics, output_path)

        return metrics

    def save_results(self, metrics: EvaluationMetrics, output_path: Path) -> None:
        """Save evaluation results to file"""
        results = {
            "evaluation_date": datetime.now().isoformat(),
            "config": asdict(self.config),
            "metrics": asdict(metrics),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to {output_path}")

    def print_summary(self, metrics: EvaluationMetrics) -> None:
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("NEXTTRACK EVALUATION RESULTS")
        print("=" * 60)

        print(
            f"Dataset: {metrics.total_users} users, {metrics.total_items} items, {metrics.total_interactions} interactions"
        )
        print(f"Evaluation time: {metrics.evaluation_time:.2f} seconds")
        print()

        print("Precision@K:")
        for k, score in metrics.precision_at_k.items():
            print(f"  P@{k}: {score:.4f}")

        print("\nRecall@K:")
        for k, score in metrics.recall_at_k.items():
            print(f"  R@{k}: {score:.4f}")

        print("\nNDCG@K:")
        for k, score in metrics.ndcg_at_k.items():
            print(f"  NDCG@{k}: {score:.4f}")

        print(f"\nOther Metrics:")
        print(f"  Coverage: {metrics.coverage:.4f}")
        print(f"  Diversity: {metrics.diversity:.4f}")
        print(f"  Novelty: {metrics.novelty:.4f}")


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description="Offline evaluation for NextTrack")
    parser.add_argument(
        "--sessions", type=Path, required=True, help="Path to sessions file"
    )
    parser.add_argument("--output", type=Path, help="Output path for results")
    parser.add_argument(
        "--max-users", type=int, help="Limit number of users for faster evaluation"
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="K values to evaluate",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train/test split ratio"
    )
    parser.add_argument(
        "--min-interactions", type=int, default=5, help="Minimum interactions per user"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Configure evaluation
    config = EvaluationConfig(
        k_values=args.k_values,
        train_ratio=args.train_ratio,
        min_interactions=args.min_interactions,
        max_users=args.max_users,
    )

    # Run evaluation
    evaluator = OfflineEvaluator(config)
    metrics = evaluator.evaluate(args.sessions, args.output)
    evaluator.print_summary(metrics)


if __name__ == "__main__":
    main()
