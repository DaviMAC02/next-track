#!/usr/bin/env python3
"""
Evaluation script for NextTrack recommendation system.
Usage: python evaluate.py --sessions data/production_sessions.json --output logs/evaluation_report.json

This script can now be run without arguments by setting sensible defaults.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluation.offline_evaluator import OfflineEvaluator, EvaluationConfig


def main_with_defaults():
    """Main evaluation script with sensible defaults"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    # Define default paths
    default_sessions = Path("data/production_sessions.json")
    default_output = Path("logs/evaluation_report.json")

    # Check dataset scale and regenerate if needed
    should_regenerate = False

    if not default_sessions.exists():
        logger.info("Sessions file not found. Will generate dataset.")
        should_regenerate = True
    else:
        # Check current dataset scale
        import json

        with open(default_sessions) as f:
            sessions = json.load(f)

        # Check if we have full scale dataset
        tracks_file = Path("data/production_tracks.json")
        if tracks_file.exists():
            with open(tracks_file) as f:
                tracks = json.load(f)

            current_tracks = len(tracks)
            current_sessions = len(sessions)

            logger.info(
                f"Current dataset: {current_tracks} tracks, {current_sessions} sessions"
            )

            # Check content features availability
            content_features_file = Path("data/metadata_features_track_ids.json")
            content_features_data_file = Path("data/metadata_features.npy")

            if (
                not content_features_file.exists()
                or not content_features_data_file.exists()
            ):
                logger.warning(
                    "Content features files missing. Will regenerate dataset."
                )
                should_regenerate = True
            else:
                # Check if content features match current tracks
                with open(content_features_file) as f:
                    content_track_ids = set(json.load(f))

                current_track_ids = set(track["track_id"] for track in tracks)

                # Check coverage
                coverage = len(content_track_ids.intersection(current_track_ids)) / len(
                    current_track_ids
                )
                logger.info(
                    f"Content features coverage: {coverage:.1%} ({len(content_track_ids)} features for {len(current_track_ids)} tracks)"
                )

                if coverage < 0.8:
                    logger.warning(
                        f"Content features coverage too low ({coverage:.1%}). Will regenerate dataset."
                    )
                    should_regenerate = True

            # Generate full dataset if needed
            if current_tracks < 500 or current_sessions < 2000:
                logger.info("Generating full-scale dataset...")
                should_regenerate = True
        else:
            should_regenerate = True

    if should_regenerate:
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/data_generator.py",
                "--tracks",
                "500",
                "--output",
                "data",
                "--users",
                "500",
                "--sessions-per-user",
                "5",
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )

        if result.returncode != 0:
            logger.error(f"Failed to generate dataset: {result.stderr}")
            logger.info("Falling back to current dataset if available...")
            if not default_sessions.exists():
                logger.error(
                    "No dataset available. Please run data generation manually."
                )
                return
        else:
            logger.info("Successfully generated dataset")

            # Run integration to build models
            logger.info("Running integration to build models...")
            integration_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/integrate_data.py",
                    "--input",
                    "data",
                    "--output",
                    "data",
                    "--models-dir",
                    "data/models",
                ],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            if integration_result.returncode != 0:
                logger.warning(
                    f"Integration step had issues: {integration_result.stderr}"
                )
            else:
                logger.info("Successfully completed integration step")

    # Ensure output directory exists
    default_output.parent.mkdir(parents=True, exist_ok=True)

    # Configure evaluation parameters
    config = EvaluationConfig(
        k_values=[5, 10, 20],
        train_ratio=0.8,
        min_interactions=5,
        max_users=500,
    )

    logger.info("Starting evaluation:")
    logger.info(f"  Sessions file: {default_sessions}")
    logger.info(f"  Output file: {default_output}")
    logger.info(f"  K values: {config.k_values}")
    logger.info(f"  Train ratio: {config.train_ratio}")
    logger.info(f"  Min interactions: {config.min_interactions}")
    logger.info(f"  Max users: {config.max_users}")

    # Run evaluation
    evaluator = OfflineEvaluator(config)
    metrics = evaluator.evaluate(default_sessions, default_output)
    evaluator.print_summary(metrics)


if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # If arguments are provided, use the original main function
        from evaluation.offline_evaluator import main

        main()
    else:
        # If no arguments, use defaults
        main_with_defaults()
