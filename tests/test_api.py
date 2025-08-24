import pytest
import json
from fastapi.testclient import TestClient
from pathlib import Path
import os

from api.main import app

# Utility to load real track IDs from generated data
TRACKS_FILE = Path("data/production_tracks.json")
TEST_TRACKS_FILE = Path("test_data/production_tracks.json")
API_KEY = os.environ.get("API_KEY", "")


def get_valid_track_ids(limit=10):
    """Get valid track IDs from production data file at test time"""
    tracks_file = TEST_TRACKS_FILE if TEST_TRACKS_FILE.exists() else TRACKS_FILE

    if not tracks_file.exists():
        # Return some fallback track IDs if no data file exists
        return ["track_1", "track_2", "track_3", "track_4", "track_5"]

    try:
        with open(tracks_file) as f:
            tracks = json.load(f)

        # Each track is a dict with 'track_id' - return first 'limit' for testing
        track_ids = [t["track_id"] for t in tracks if "track_id" in t][:limit]

        if not track_ids:
            # Fallback if no valid track IDs found
            return ["track_1", "track_2", "track_3", "track_4", "track_5"]

        return track_ids
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        # Fallback on any error
        return ["track_1", "track_2", "track_3", "track_4", "track_5"]


@pytest.fixture(scope="session")
def real_track_ids():
    # Use helper function to get valid track IDs
    return get_valid_track_ids()


@pytest.fixture(scope="module")
def client():
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    with TestClient(app, headers=headers) as c:
        yield c


class TestNextTrackAPI:
    def test_health_check(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json() or "status" in response.json()

    def test_detailed_health_check(self, client):
        response = client.get("/health")
        assert response.status_code in [200]

    def test_get_recommendations_basic(self, client):
        # Get fresh track IDs at test time
        track_ids = get_valid_track_ids(5)
        request_data = {"track_sequence": track_ids[:2], "top_k": 5}
        response = client.post("/recommendations", json=request_data)
        if response.status_code not in [200]:
            print("/recommendations response:", response.text)
        assert response.status_code in [200]
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert "scores" in data
            assert "metadata" in data
            assert len(data["recommendations"]) <= 5

    def test_get_recommendations_with_filters(self, client):
        # Get fresh track IDs at test time
        track_ids = get_valid_track_ids(5)
        request_data = {
            "track_sequence": track_ids[:2],
            "top_k": 3,
            "genre_filter": ["rock", "pop"],
            "mood_filter": ["happy"],
            "fusion_method": "weighted_sum",
        }
        response = client.post("/recommendations", json=request_data)
        if response.status_code not in [200]:
            print("/recommendations (filters) response:", response.text)
        assert response.status_code in [200]

    def test_legacy_next_track_endpoint(self, client):
        # Get fresh track IDs at test time
        track_ids = get_valid_track_ids(3)
        # Send as query params, not JSON
        params = [("track_sequence", tid) for tid in track_ids[:1]]
        params.append(("top_k", 3))
        response = client.post("/next_track", params=params)
        if response.status_code not in [200]:
            print("/next_track response:", response.text)
        assert response.status_code in [200]

    def test_similar_tracks(self, client):
        # Get fresh track IDs at test time
        track_ids = get_valid_track_ids(3)
        request_data = {
            "track_id": track_ids[0],
            "top_k": 5,
            "use_cf": True,
            "use_cb": True,
        }
        response = client.post("/similar_tracks", json=request_data)
        assert response.status_code in [200]

    def test_system_stats(self, client):
        response = client.get("/stats")
        if response.status_code != 200:
            print("/stats response:", response.text)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_list_tracks(self, client):
        response = client.get("/tracks")
        if response.status_code not in [
            200,
        ]:
            print("/tracks response:", response.text)
        assert response.status_code in [
            200,
        ]
        if response.status_code == 200:
            data = response.json()
            assert "tracks" in data
            assert "total" in data

    def test_list_tracks_pagination(self, client):
        response = client.get("/tracks?limit=5&offset=0")
        if response.status_code not in [
            200,
        ]:
            print("/tracks (pagination) response:", response.text)
        assert response.status_code in [
            200,
        ]

    def test_invalid_fusion_method(self, client):
        # Get fresh track IDs at test time
        track_ids = get_valid_track_ids(2)
        request_data = {
            "track_sequence": track_ids[:1],
            "fusion_method": "invalid_method",
        }
        response = client.post("/recommendations", json=request_data)
        assert response.status_code == 422

    def test_empty_track_sequence(self, client):
        request_data = {"track_sequence": [], "top_k": 5}
        response = client.post("/recommendations", json=request_data)
        assert response.status_code == 422

    def test_invalid_top_k(self, client):
        # Get fresh track IDs at test time
        track_ids = get_valid_track_ids(2)
        # Too small
        request_data = {"track_sequence": track_ids[:1], "top_k": 0}
        response = client.post("/recommendations", json=request_data)
        assert response.status_code == 422
        # Too large
        request_data["top_k"] = 1000
        response = client.post("/recommendations", json=request_data)
        assert response.status_code == 422

    def test_unknown_track_id(self, client):
        request_data = {"track_sequence": ["unknown_track"], "top_k": 5}
        response = client.post("/recommendations", json=request_data)
        if response.status_code not in [400]:
            print("/recommendations (unknown_track) response:", response.text)
        assert response.status_code in [400]

    # Monitoring Endpoints Tests
    def test_monitoring_health(self, client):
        response = client.get("/monitoring/health")
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert (
                "checks" in data
                or "components" in data
                or "status" in data
                or "overall_status" in data
            )

    def test_monitoring_metrics(self, client):
        response = client.get("/monitoring/metrics")
        assert response.status_code in [200]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_monitoring_prometheus(self, client):
        response = client.get("/monitoring/prometheus")
        assert response.status_code in [200, 501]

    def test_monitoring_cache_stats(self, client):
        response = client.get("/monitoring/cache")
        assert response.status_code in [200]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_monitoring_cache_clear(self, client):
        response = client.post("/monitoring/cache/clear")
        assert response.status_code in [200]
        if response.status_code == 200:
            data = response.json()
            assert "success" in data

    def test_monitoring_cache_invalidate_pattern(self, client):
        response = client.delete("/monitoring/cache/test_pattern")
        assert response.status_code in [200]
        if response.status_code == 200:
            data = response.json()
            assert "invalidated_count" in data

    # Feedback Endpoints Tests
    def test_feedback_submit(self, client):
        feedback_data = {
            "session_id": "test_session_123",
            "user_id": "test_user_456",
            "track_id": "test_track_789",
            "feedback_type": "like",
            "feedback_value": None,
        }
        response = client.post("/feedback", json=feedback_data)
        assert response.status_code in [200]
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"

    def test_feedback_stats(self, client):
        response = client.get("/feedback/stats")
        assert response.status_code in [200]
        if response.status_code == 200:
            data = response.json()
            assert "feedback_stats" in data
            assert "timestamp" in data

    def test_feedback_track_specific(self, client):
        # Get valid track IDs at test time
        track_ids = get_valid_track_ids(3)
        response = client.get(f"/feedback/track/{track_ids[0]}")
        assert response.status_code in [200]
        if response.status_code == 200:
            data = response.json()
            assert "track_id" in data
            assert "feedback_count" in data

    def test_feedback_track_specific_with_days(self, client):
        # Get valid track IDs at test time
        track_ids = get_valid_track_ids(3)
        response = client.get(f"/feedback/track/{track_ids[0]}?days=30")
        assert response.status_code in [200]
        if response.status_code == 200:
            data = response.json()
            assert "days" in data
            assert data["days"] == 30

    # Experiments Endpoints Tests
    def test_experiments_create(self, client):
        experiment_data = {
            "name": "Test CF Weight Experiment",
            "description": "Testing different collaborative filtering weights",
            "traffic_split": 0.1,
            "config_overrides": {"cf_weight": 0.8, "cb_weight": 0.2},
            "success_metrics": ["precision", "engagement"],
            "duration_days": 14,
        }
        response = client.post("/experiments", json=experiment_data)
        assert response.status_code in [200]
        if response.status_code == 200:
            data = response.json()
            assert "experiment_id" in data
            assert data["status"] == "created"

    def test_experiments_list(self, client):
        response = client.get("/experiments")
        assert response.status_code in [200]
        if response.status_code == 200:
            data = response.json()
            assert "experiments" in data
            assert "total_count" in data
            assert isinstance(data["experiments"], list)

    def test_experiments_start(self, client):
        # First create an experiment
        experiment_data = {
            "name": "Test Start Experiment",
            "description": "Testing experiment start functionality",
            "traffic_split": 0.05,
            "config_overrides": {"cf_weight": 0.9},
            "success_metrics": ["ctr"],
            "duration_days": 7,
        }
        create_response = client.post("/experiments", json=experiment_data)
        if create_response.status_code == 200:
            experiment_id = create_response.json()["experiment_id"]

            # Now start the experiment
            response = client.post(f"/experiments/{experiment_id}/start")
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                data = response.json()
                assert data["status"] == "started"

    def test_experiments_stop(self, client):
        # First create and start an experiment
        experiment_data = {
            "name": "Test Stop Experiment",
            "description": "Testing experiment stop functionality",
            "traffic_split": 0.05,
            "config_overrides": {"cb_weight": 0.7},
            "success_metrics": ["recall"],
            "duration_days": 7,
        }
        create_response = client.post("/experiments", json=experiment_data)
        if create_response.status_code == 200:
            experiment_id = create_response.json()["experiment_id"]

            # Start the experiment
            client.post(f"/experiments/{experiment_id}/start")

            # Now stop the experiment
            response = client.post(f"/experiments/{experiment_id}/stop")
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                data = response.json()
                assert data["status"] == "stopped"

    def test_experiments_start_nonexistent(self, client):
        response = client.post("/experiments/nonexistent_id/start")
        assert response.status_code == 404

    def test_experiments_stop_nonexistent(self, client):
        response = client.post("/experiments/nonexistent_id/stop")
        assert response.status_code == 404

    # Additional Edge Case Tests
    def test_feedback_invalid_type(self, client):
        feedback_data = {
            "session_id": "test_session",
            "track_id": "test_track",
            "feedback_type": "invalid_type",
        }
        response = client.post("/feedback", json=feedback_data)
        assert response.status_code == 422

    def test_experiments_invalid_traffic_split(self, client):
        experiment_data = {
            "name": "Invalid Traffic Split Test",
            "description": "Testing invalid traffic split",
            "traffic_split": 1.5,  # Invalid: > 1.0
            "config_overrides": {},
            "success_metrics": ["ctr"],
        }
        response = client.post("/experiments", json=experiment_data)
        assert response.status_code == 422
