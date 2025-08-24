#!/usr/bin/env python3
"""
NextTrack API - Manual Testing Script
=====================================

This script provides a comprehensive test suite for the NextTrack API.
Run this script to verify all functionality is working correctly.

Usage:
    python manual_test_script.py

Requirements:
    - Python 3.10+
    - All requirements.txt dependencies installed
    - Virtual environment activated
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_step(step_num: int, description: str):
    """Print a formatted test step"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print("=" * 60)


def print_success(message: str):
    """Print a success message"""
    print(f"✅ {message}")


def print_error(message: str):
    """Print an error message"""
    print(f"❌ {message}")


def print_info(message: str):
    """Print an info message"""
    print(f"ℹ️  {message}")


def test_data_generation():
    """Test Step 1: Data Generation from MusicBrainz"""
    print_step(1, "Testing Data Generation from MusicBrainz API")

    try:
        # Check if we can import the data generator
        from scripts.data_generator import MusicBrainzDataGenerator

        print_success("Data generator imports successfully")

        # Test MusicBrainz connection
        print_info("Testing MusicBrainz API connection...")
        generator = MusicBrainzDataGenerator()
        test_tracks = generator.search_tracks("rock", limit=1)

        if test_tracks:
            print_success(
                f"MusicBrainz API connection working - found {len(test_tracks)} tracks"
            )
            track = test_tracks[0]
            print_info(f"Sample track: \"{track.get('title', 'Unknown')}\"")
        else:
            print_error("MusicBrainz API connection failed or no tracks found")
            return False

    except ImportError as e:
        print_error(f"Failed to import data generator: {e}")
        return False
    except Exception as e:
        print_error(f"Data generation test failed: {e}")
        return False

    return True


def test_full_data_pipeline():
    """Test Step 2: Full Data Generation Pipeline"""
    print_step(2, "Testing Full Data Generation Pipeline")

    try:
        import subprocess

        print_info("Running data generation script for 10 tracks...")
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate.py",
                "--tracks",
                "10",
                "--output",
                "test_data",
                "--models-dir",
                "test_data/models",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        if result.returncode == 0:
            print_success("Data generation pipeline completed successfully")

            # Check generated files
            test_data_dir = Path("test_data")
            expected_files = [
                "production_tracks.json",
                "production_track_metadata.json",
                "production_sessions.json",
                "production_training_sessions.json",
                "lightfm_embeddings.json",
            ]

            for file in expected_files:
                file_path = test_data_dir / file
                if file_path.exists():
                    print_success(f"Generated file: {file}")
                else:
                    print_error(f"Missing file: {file}")
                    return False

            # Check models directory
            models_dir = test_data_dir / "models"
            expected_model_files = [
                "cb_features.pkl",
                "lightfm_model.pkl",
                "training_sessions.pkl",
            ]

            for file in expected_model_files:
                file_path = models_dir / file
                if file_path.exists():
                    print_success(f"Generated model: {file}")
                else:
                    print_error(f"Missing model: {file}")
                    return False

        else:
            print_error(f"Data generation failed: {result.stderr}")
            return False

    except Exception as e:
        print_error(f"Data pipeline test failed: {e}")
        return False

    return True


def test_api_startup():
    """Test Step 3: API Startup and Health Check"""
    print_step(3, "Testing API Startup and Health Check")

    try:
        # Set up test environment
        os.environ["USE_ANN_INDEX"] = "false"
        os.environ["USE_CONTENT_BASED"] = "true"
        os.environ["DATA_DIR"] = "test_data"

        # Import and test API
        from api.main import app
        from fastapi.testclient import TestClient

        print_info("Starting API with test client...")
        with TestClient(app) as client:
            # Test health endpoint
            response = client.get("/health")
            if response.status_code == 200:
                health_data = response.json()
                print_success(
                    f"API Health Check: {health_data.get('status', 'unknown')}"
                )
                print_info(f"API Version: {health_data.get('version', 'unknown')}")
                print_info(f"Environment: {health_data.get('environment', 'unknown')}")
            else:
                print_error(f"Health check failed with status {response.status_code}")
                return False

            # Test API documentation
            response = client.get("/docs")
            if response.status_code == 200:
                print_success("Swagger UI documentation is accessible")
            else:
                print_error("Swagger UI documentation is not accessible")
                return False

    except Exception as e:
        print_error(f"API startup test failed: {e}")
        return False

    return True


def test_recommendation_functionality():
    """Test Step 4: Core Recommendation Functionality"""
    print_step(4, "Testing Core Recommendation Functionality")

    try:
        # Set up environment
        os.environ["USE_ANN_INDEX"] = "false"
        os.environ["USE_CONTENT_BASED"] = "true"
        os.environ["DATA_DIR"] = "test_data"

        from api.main import app
        from fastapi.testclient import TestClient

        print_info("Testing recommendation endpoints...")
        with TestClient(app) as client:
            # Load test tracks
            with open("test_data/production_tracks.json", "r") as f:
                tracks = json.load(f)

            if len(tracks) < 2:
                print_error("Not enough tracks for testing recommendations")
                return False

            track_ids = [track["track_id"] for track in tracks[:2]]
            print_info(f"Using tracks: {[track['title'] for track in tracks[:2]]}")

            # Test recommendations endpoint
            rec_request = {
                "track_sequence": track_ids,
                "top_k": 5,
                "fusion_method": "weighted_sum",
            }

            response = client.post("/recommendations", json=rec_request)
            if response.status_code == 200:
                rec_data = response.json()
                recommendations = rec_data.get("recommendations", [])
                print_success(
                    f"Recommendations received: {len(recommendations)} tracks"
                )
                print_info(
                    f"Fusion method: {rec_data.get('metadata', {}).get('fusion_method')}"
                )
                print_info(
                    f"CF weight: {rec_data.get('metadata', {}).get('cf_weight')}"
                )
                print_info(
                    f"CB weight: {rec_data.get('metadata', {}).get('cb_weight')}"
                )
            else:
                print_error(
                    f"Recommendations failed with status {response.status_code}"
                )
                print_error(f"Response: {response.text}")
                return False

            # Test track listing
            response = client.get("/tracks?page=1&page_size=5")
            if response.status_code == 200:
                tracks_data = response.json()
                print_success(
                    f"Track listing: {len(tracks_data.get('tracks', []))} tracks retrieved"
                )
            else:
                print_error(f"Track listing failed with status {response.status_code}")
                return False

    except Exception as e:
        print_error(f"Recommendation functionality test failed: {e}")
        return False

    return True


def test_api_endpoints():
    """Test Step 5: Additional API Endpoints"""
    print_step(5, "Testing Additional API Endpoints")

    try:
        os.environ["USE_ANN_INDEX"] = "false"
        os.environ["USE_CONTENT_BASED"] = "true"
        os.environ["DATA_DIR"] = "test_data"

        from api.main import app
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            # Test metrics endpoint
            response = client.get("/metrics")
            if response.status_code == 200:
                print_success("Metrics endpoint is working")
            else:
                print_info(
                    f"Metrics endpoint returned {response.status_code} (may not be enabled)"
                )

            # Test OpenAPI spec
            response = client.get("/openapi.json")
            if response.status_code == 200:
                openapi_spec = response.json()
                print_success(
                    f"OpenAPI specification available with {len(openapi_spec.get('paths', {}))} endpoints"
                )
            else:
                print_error("OpenAPI specification not available")
                return False

            # Test ReDoc documentation
            response = client.get("/redoc")
            if response.status_code == 200:
                print_success("ReDoc documentation is accessible")
            else:
                print_info("ReDoc documentation may not be available")

    except Exception as e:
        print_error(f"API endpoints test failed: {e}")
        return False

    return True


def test_unit_tests():
    """Test Step 6: Run Unit Test Suite"""
    print_step(6, "Running Unit Test Suite")

    try:
        import subprocess

        print_info("Running pytest test suite...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_api.py", "-v"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        if result.returncode == 0:
            # Count passed tests
            output_lines = result.stdout.split("\n")
            passed_line = [
                line for line in output_lines if "passed" in line and "warnings" in line
            ]
            if passed_line:
                print_success(f"Unit tests completed: {passed_line[0]}")
            else:
                print_success("Unit tests completed successfully")
        else:
            print_error("Some unit tests failed")
            print_error(f"Test output: {result.stdout}")
            print_error(f"Test errors: {result.stderr}")
            return False

    except Exception as e:
        print_error(f"Unit test execution failed: {e}")
        return False

    return True


def cleanup_test_data():
    """Clean up temporary test data directory"""
    print_step(7, "Cleaning Up Test Data")

    try:
        import shutil

        test_data_dir = Path("test_data")
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)
            print_success("Test data cleaned up successfully")
        else:
            print_info("No test data to clean up")
    except Exception as e:
        print_error(f"Failed to clean up test data: {e}")


def main():
    """Main test execution function"""
    print("🎉 NextTrack API - Manual Test Suite")
    print("=" * 60)
    print("This script will test all major functionality of the NextTrack API.")
    print("Make sure you have:")
    print("  1. Python virtual environment activated")
    print("  2. All requirements.txt dependencies installed")
    print("  3. Internet connection for MusicBrainz API")
    print("=" * 60)

    input("Press Enter to start testing...")

    # Track test results
    test_results = []

    # Run all tests
    tests = [
        ("Data Generation", test_data_generation),
        ("Full Data Pipeline", test_full_data_pipeline),
        ("API Startup", test_api_startup),
        ("Recommendation Functionality", test_recommendation_functionality),
        ("API Endpoints", test_api_endpoints),
        ("Unit Tests", test_unit_tests),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
            if result:
                print_success(f"{test_name} - PASSED")
            else:
                print_error(f"{test_name} - FAILED")
        except Exception as e:
            print_error(f"{test_name} - ERROR: {e}")
            test_results.append((test_name, False))

    # Clean up
    cleanup_test_data()

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎊 ALL TESTS PASSED! NextTrack API is fully functional!")
        print("✨ The system is ready for:")
        print("   • Production deployment")
        print("   • Presentation demos")
        print("   • Real-world music recommendations")
    else:
        print(
            f"\n⚠️  {total - passed} test(s) failed. Please check the output above for details."
        )

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
