"""
Tests for the offline evaluation system.
"""

import pytest
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.offline_evaluator import (
    OfflineEvaluator,
    EvaluationConfig,
    EvaluationMetrics,
)


@pytest.fixture
def sample_sessions():
    """Sample session data for testing"""
    return [
        {
            "session_id": "session_1",
            "user_id": "user_1",
            "track_sequence": ["track_1", "track_2", "track_3", "track_4", "track_5"],
        },
        {
            "session_id": "session_2",
            "user_id": "user_2",
            "track_sequence": ["track_2", "track_3", "track_4", "track_5", "track_6"],
        },
        {
            "session_id": "session_3",
            "user_id": "user_3",
            "track_sequence": [
                "track_1",
                "track_3",
                "track_5",
                "track_7",
                "track_8",
                "track_9",
            ],
        },
    ]


@pytest.fixture
def temp_sessions_file(sample_sessions):
    """Temporary sessions file for testing"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_sessions, f)
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def evaluator():
    """OfflineEvaluator instance with test configuration"""
    config = EvaluationConfig(
        k_values=[1, 3, 5],
        train_ratio=0.6,  # 60% train, 40% test
        min_interactions=5,
        max_users=10,
        random_seed=42,
    )
    return OfflineEvaluator(config)


class TestEvaluationConfig:
    """Test evaluation configuration"""

    def test_default_config(self):
        config = EvaluationConfig()
        assert config.k_values == [1, 3, 5, 10, 20]
        assert config.train_ratio == 0.8
        assert config.min_interactions == 5
        assert config.random_seed == 42

    def test_custom_config(self):
        config = EvaluationConfig(
            k_values=[1, 5, 10], train_ratio=0.7, min_interactions=3, max_users=100
        )
        assert config.k_values == [1, 5, 10]
        assert config.train_ratio == 0.7
        assert config.min_interactions == 3
        assert config.max_users == 100


class TestOfflineEvaluator:
    """Test offline evaluation functionality"""

    def test_load_sessions(self, evaluator, temp_sessions_file):
        sessions = evaluator.load_sessions(temp_sessions_file)
        assert len(sessions) == 3
        assert sessions[0]["session_id"] == "session_1"
        assert len(sessions[0]["track_sequence"]) == 5

    def test_preprocess_sessions(self, evaluator, sample_sessions):
        # All sessions have >= 5 tracks, so all should be kept
        processed = evaluator.preprocess_sessions(sample_sessions)
        assert len(processed) == 3

        # Test filtering by min_interactions
        evaluator.config.min_interactions = 6
        processed = evaluator.preprocess_sessions(sample_sessions)
        assert len(processed) == 1  # Only session_3 has 6 tracks

    def test_train_test_split(self, evaluator, sample_sessions):
        train_sessions, test_sessions = evaluator.train_test_split(sample_sessions)

        # Should have some train and test sessions (session-level split)
        assert len(train_sessions) > 0
        assert len(test_sessions) > 0
        assert len(train_sessions) + len(test_sessions) <= len(sample_sessions)

        # Check that test sessions have ground truth
        for test_session in test_sessions:
            assert "ground_truth" in test_session
            assert len(test_session["ground_truth"]) > 0
            assert len(test_session["track_sequence"]) > 0

    def test_compute_precision_recall(self, evaluator):
        recommendations = ["track_1", "track_2", "track_3", "track_4", "track_5"]
        ground_truth = ["track_1", "track_3", "track_6", "track_7"]

        # Test precision@3 and recall@3
        precision, recall = evaluator.compute_precision_recall(
            recommendations, ground_truth, k=3
        )

        # Recommendations[:3] = ["track_1", "track_2", "track_3"]
        # Intersection with ground_truth = ["track_1", "track_3"] = 2 hits
        # Precision@3 = 2/3 ≈ 0.667
        # Recall@3 = 2/4 = 0.5
        assert abs(precision - 2 / 3) < 0.001
        assert recall == 0.5

    def test_compute_ndcg(self, evaluator):
        recommendations = ["track_1", "track_2", "track_3"]
        ground_truth = ["track_1", "track_3"]

        ndcg = evaluator.compute_ndcg(recommendations, ground_truth, k=3)

        # DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.631 = 1.631
        # NDCG = 1.5 / 1.631 ≈ 0.92
        assert 0.9 < ndcg < 1.0

    def test_compute_coverage(self, evaluator):
        all_recommendations = [
            ["track_1", "track_2", "track_3"],
            ["track_2", "track_4", "track_5"],
            ["track_1", "track_6", "track_7"],
        ]
        catalog_size = 10

        coverage = evaluator.compute_coverage(all_recommendations, catalog_size)

        # Unique tracks recommended: {track_1, track_2, track_3, track_4, track_5, track_6, track_7} = 7
        # Coverage = 7/10 = 0.7
        assert coverage == 0.7

    def test_compute_novelty(self, evaluator):
        recommendations = [["track_1", "track_2"], ["track_3", "track_4"]]
        item_popularity = {
            "track_1": 100,  # Very popular
            "track_2": 10,  # Less popular
            "track_3": 1,  # Rare
            "track_4": 50,  # Moderately popular
        }

        novelty = evaluator.compute_novelty(recommendations, item_popularity)

        # Higher novelty expected for less popular items
        assert novelty > 0


class TestEvaluationMetrics:
    """Test evaluation metrics data structure"""

    def test_metrics_creation(self):
        metrics = EvaluationMetrics(
            precision_at_k={1: 0.1, 5: 0.05},
            recall_at_k={1: 0.2, 5: 0.15},
            ndcg_at_k={1: 0.15, 5: 0.12},
            coverage=0.3,
            diversity=0.7,
            novelty=5.2,
            total_users=100,
            total_items=1000,
            total_interactions=5000,
            evaluation_time=120.5,
        )

        assert metrics.precision_at_k[1] == 0.1
        assert metrics.total_users == 100
        assert metrics.evaluation_time == 120.5


class TestEvaluationIntegration:
    """Integration tests for the complete evaluation pipeline"""

    @patch("evaluation.offline_evaluator.HybridRecommender")
    def test_evaluation_pipeline_mock(
        self, mock_recommender_class, evaluator, temp_sessions_file
    ):
        """Test evaluation pipeline with mocked recommender"""

        # Mock the recommender
        mock_recommender = Mock()
        mock_recommender._initialized = True
        mock_recommender.get_recommendations.return_value = {
            "recommendations": ["track_6", "track_7", "track_8"],
            "scores": [0.9, 0.8, 0.7],
            "metadata": {"cf_available": True, "cb_available": True},
        }
        mock_recommender.vectorizer.get_embeddings_matrix.return_value = np.random.rand(
            10, 64
        )
        mock_recommender.vectorizer.get_track_ids.return_value = [
            f"track_{i}" for i in range(1, 11)
        ]

        mock_recommender_class.return_value = mock_recommender

        # Mock settings
        with patch("evaluation.offline_evaluator.settings") as mock_settings:
            mock_settings.CF_WEIGHT = 0.7
            mock_settings.CB_WEIGHT = 0.3
            mock_settings.USE_CONTENT_BASED = True
            mock_settings.METADATA_FEATURES_FILE.exists.return_value = True
            mock_settings.USE_ANN_INDEX = False
            mock_settings.EMBEDDINGS_FILE = Path("test_embeddings.json")
            mock_settings.METADATA_FEATURES_FILE = Path("test_features.npy")

            # Run evaluation
            metrics = evaluator.evaluate(temp_sessions_file)

            # Check metrics were computed
            assert isinstance(metrics, EvaluationMetrics)
            assert len(metrics.precision_at_k) == len(evaluator.config.k_values)
            assert len(metrics.recall_at_k) == len(evaluator.config.k_values)
            assert len(metrics.ndcg_at_k) == len(evaluator.config.k_values)
            assert metrics.total_users > 0
            assert metrics.evaluation_time > 0

    def test_save_and_load_results(self, evaluator):
        """Test saving evaluation results to file"""
        metrics = EvaluationMetrics(
            precision_at_k={1: 0.1, 5: 0.05},
            recall_at_k={1: 0.2, 5: 0.15},
            ndcg_at_k={1: 0.15, 5: 0.12},
            coverage=0.3,
            diversity=0.7,
            novelty=5.2,
            total_users=100,
            total_items=1000,
            total_interactions=5000,
            evaluation_time=120.5,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            evaluator.save_results(metrics, output_path)

            # Check file was created and has correct content
            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            assert "evaluation_date" in data
            assert "config" in data
            assert "metrics" in data
            assert data["metrics"]["total_users"] == 100
            assert data["metrics"]["precision_at_k"]["1"] == 0.1

        finally:
            # Cleanup
            if output_path.exists():
                output_path.unlink()

    def test_print_summary(self, evaluator, capsys):
        """Test evaluation summary printing"""
        metrics = EvaluationMetrics(
            precision_at_k={1: 0.1, 5: 0.05},
            recall_at_k={1: 0.2, 5: 0.15},
            ndcg_at_k={1: 0.15, 5: 0.12},
            coverage=0.3,
            diversity=0.7,
            novelty=5.2,
            total_users=100,
            total_items=1000,
            total_interactions=5000,
            evaluation_time=120.5,
        )

        evaluator.print_summary(metrics)

        captured = capsys.readouterr()
        assert "NEXTTRACK EVALUATION RESULTS" in captured.out
        assert "P@1: 0.1000" in captured.out
        assert "R@5: 0.1500" in captured.out
        assert "Coverage: 0.3000" in captured.out
