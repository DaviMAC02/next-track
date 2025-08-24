"""
A/B Testing and User Feedback System for NextTrack API.
Enables experimentation with different recommendation strategies and collection of user feedback.
"""

import json
import logging
import time
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import random
from enum import Enum

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback"""

    LIKE = "like"
    DISLIKE = "dislike"
    SKIP = "skip"
    PLAY_COMPLETE = "play_complete"
    EXPLICIT_RATING = "explicit_rating"


class ExperimentStatus(Enum):
    """A/B experiment status"""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class FeedbackEvent:
    """User feedback event"""

    session_id: str
    user_id: Optional[str]
    track_id: str
    feedback_type: FeedbackType
    feedback_value: Optional[float]  # For ratings (1-5) or play duration
    timestamp: float
    experiment_id: Optional[str] = None
    recommendation_context: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentConfig:
    """A/B experiment configuration"""

    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    start_date: datetime
    end_date: Optional[datetime]
    traffic_split: float  # Percentage of traffic (0.0 to 1.0)
    config_overrides: Dict[str, Any]  # Configuration changes for this experiment
    success_metrics: List[str]  # Metrics to track ['ctr', 'engagement', 'diversity']


@dataclass
class ABTestResult:
    """A/B test results"""

    experiment_id: str
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    statistical_significance: Dict[str, float]  # p-values
    sample_sizes: Dict[str, int]
    confidence_intervals: Dict[str, tuple]


class FeedbackCollector:
    """Collects and stores user feedback"""

    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path("logs/feedback.jsonl")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory storage for recent feedback
        self.recent_feedback: deque = deque(maxlen=1000)
        self.feedback_stats = defaultdict(int)

    def record_feedback(self, feedback: FeedbackEvent) -> None:
        """Record a feedback event"""
        # Add to recent feedback
        self.recent_feedback.append(feedback)

        # Update stats
        self.feedback_stats[f"total_{feedback.feedback_type.value}"] += 1
        self.feedback_stats["total_events"] += 1

        # Persist to file (convert enum to string for JSON serialization)
        feedback_dict = asdict(feedback)
        feedback_dict["feedback_type"] = feedback.feedback_type.value

        with open(self.storage_path, "a") as f:
            f.write(json.dumps(feedback_dict) + "\n")

        logger.info(
            f"Recorded feedback: {feedback.feedback_type.value} for track {feedback.track_id}"
        )

    def get_track_feedback(self, track_id: str, days: int = 7) -> List[FeedbackEvent]:
        """Get recent feedback for a specific track"""
        cutoff_time = time.time() - (days * 24 * 3600)

        track_feedback = []
        for feedback in self.recent_feedback:
            if feedback.track_id == track_id and feedback.timestamp > cutoff_time:
                track_feedback.append(feedback)

        return track_feedback

    def get_user_feedback(self, user_id: str, days: int = 30) -> List[FeedbackEvent]:
        """Get recent feedback for a specific user"""
        cutoff_time = time.time() - (days * 24 * 3600)

        user_feedback = []
        for feedback in self.recent_feedback:
            if feedback.user_id == user_id and feedback.timestamp > cutoff_time:
                user_feedback.append(feedback)

        return user_feedback

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        total_events = self.feedback_stats.get("total_events", 0)

        if total_events == 0:
            return {"total_events": 0}

        stats = dict(self.feedback_stats)

        # Calculate rates
        stats["like_rate"] = stats.get("total_like", 0) / total_events
        stats["dislike_rate"] = stats.get("total_dislike", 0) / total_events
        stats["skip_rate"] = stats.get("total_skip", 0) / total_events
        stats["completion_rate"] = stats.get("total_play_complete", 0) / total_events

        return stats


class ABTester:
    """A/B testing framework for recommendation experiments"""

    def __init__(self, experiments_path: Path = None):
        self.experiments_path = experiments_path or Path("config/experiments.json")
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.load_experiments()

        # User assignment cache
        self.user_assignments: Dict[str, str] = {}  # user_id -> experiment_id

    def load_experiments(self) -> None:
        """Load experiment configurations"""
        if self.experiments_path.exists():
            try:
                with open(self.experiments_path) as f:
                    data = json.load(f)

                for exp_data in data.get("experiments", []):
                    exp = ExperimentConfig(
                        experiment_id=exp_data["experiment_id"],
                        name=exp_data["name"],
                        description=exp_data["description"],
                        status=ExperimentStatus(exp_data["status"]),
                        start_date=datetime.fromisoformat(exp_data["start_date"]),
                        end_date=(
                            datetime.fromisoformat(exp_data["end_date"])
                            if exp_data.get("end_date")
                            else None
                        ),
                        traffic_split=exp_data["traffic_split"],
                        config_overrides=exp_data["config_overrides"],
                        success_metrics=exp_data["success_metrics"],
                    )
                    self.experiments[exp.experiment_id] = exp

                logger.info(f"Loaded {len(self.experiments)} experiments")
            except Exception as e:
                logger.error(f"Failed to load experiments: {e}")

    def save_experiments(self) -> None:
        """Save experiment configurations"""
        data = {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "description": exp.description,
                    "status": exp.status.value,
                    "start_date": exp.start_date.isoformat(),
                    "end_date": exp.end_date.isoformat() if exp.end_date else None,
                    "traffic_split": exp.traffic_split,
                    "config_overrides": exp.config_overrides,
                    "success_metrics": exp.success_metrics,
                }
                for exp in self.experiments.values()
            ]
        }

        self.experiments_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.experiments_path, "w") as f:
            json.dump(data, f, indent=2)

    def assign_user_to_experiment(self, user_id: str) -> Optional[str]:
        """Assign user to an active experiment"""
        # Check if user already assigned
        if user_id in self.user_assignments:
            exp_id = self.user_assignments[user_id]
            if (
                exp_id in self.experiments
                and self.experiments[exp_id].status == ExperimentStatus.ACTIVE
            ):
                return exp_id

        # Find active experiments
        active_experiments = [
            exp
            for exp in self.experiments.values()
            if exp.status == ExperimentStatus.ACTIVE
            and exp.start_date <= datetime.now()
            and (exp.end_date is None or exp.end_date >= datetime.now())
        ]

        if not active_experiments:
            return None

        # Deterministic assignment based on user ID hash
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)

        # Check traffic allocation for each experiment
        for exp in active_experiments:
            # Use hash to determine if user is in this experiment's traffic
            if (user_hash % 100) / 100.0 < exp.traffic_split:
                self.user_assignments[user_id] = exp.experiment_id
                logger.info(
                    f"Assigned user {user_id} to experiment {exp.experiment_id}"
                )
                return exp.experiment_id

        return None

    def get_experiment_config(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration overrides for an experiment"""
        if experiment_id in self.experiments:
            return self.experiments[experiment_id].config_overrides
        return None

    def create_experiment(
        self,
        name: str,
        description: str,
        traffic_split: float,
        config_overrides: Dict[str, Any],
        success_metrics: List[str],
        duration_days: int = 30,
    ) -> str:
        """Create a new A/B experiment"""
        experiment_id = f"exp_{int(time.time())}"
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)

        experiment = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT,
            start_date=start_date,
            end_date=end_date,
            traffic_split=traffic_split,
            config_overrides=config_overrides,
            success_metrics=success_metrics,
        )

        self.experiments[experiment_id] = experiment
        self.save_experiments()

        logger.info(f"Created experiment {experiment_id}: {name}")
        return experiment_id

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].status = ExperimentStatus.ACTIVE
            self.save_experiments()
            logger.info(f"Started experiment {experiment_id}")
            return True
        return False

    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id].status = ExperimentStatus.COMPLETED
            self.save_experiments()
            logger.info(f"Stopped experiment {experiment_id}")
            return True
        return False


class ExperimentalRecommender:
    """Wrapper around HybridRecommender that applies experimental configurations"""

    def __init__(self, base_recommender, ab_tester: ABTester):
        self.base_recommender = base_recommender
        self.ab_tester = ab_tester

    def get_recommendations(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Get recommendations with experimental configuration"""
        # Determine experiment assignment
        experiment_id = self.ab_tester.assign_user_to_experiment(user_id)

        # Apply experimental config
        if experiment_id:
            exp_config = self.ab_tester.get_experiment_config(experiment_id)
            if exp_config:
                # Override parameters with experimental values
                for key, value in exp_config.items():
                    if key in kwargs:
                        kwargs[key] = value

        # Get recommendations from base recommender
        result = self.base_recommender.get_recommendations(**kwargs)

        # Add experiment metadata
        result["experiment_id"] = experiment_id
        result["experiment_config"] = (
            self.ab_tester.get_experiment_config(experiment_id)
            if experiment_id
            else None
        )

        return result


# Global instances
feedback_collector = FeedbackCollector()
ab_tester = ABTester()


def get_feedback_collector() -> FeedbackCollector:
    """Get the global feedback collector"""
    return feedback_collector


def get_ab_tester() -> ABTester:
    """Get the global A/B tester"""
    return ab_tester
