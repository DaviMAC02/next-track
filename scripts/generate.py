#!/usr/bin/env python3
"""
Data Material Generation Script

Runs the data generator and integration pipeline in sequence to produce all data and models needed for the API.
"""
import sys
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Generate all data and models for NextTrack API"
    )
    parser.add_argument(
        "--tracks", type=int, default=500, help="Number of tracks to generate"
    )
    parser.add_argument(
        "--output", type=str, default="data", help="Directory to store generated data"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models",
        help="Directory to store trained models",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"🏗️  Generating {args.tracks} tracks and all data material...")
    print("=" * 80)

    # 1. Run data generator
    subprocess.run(
        [
            sys.executable,
            "scripts/data_generator.py",
            "--tracks",
            str(args.tracks),
            "--output",
            args.output,
        ],
        check=True,
    )

    # 2. Run integration
    subprocess.run(
        [
            sys.executable,
            "scripts/integrate_data.py",
            "--data-dir",
            args.output,
            "--output-dir",
            args.models_dir,
        ],
        check=True,
    )

    print("\n✅ Data and models generated successfully!")
    print(f"  Data directory: {args.output}")
    print(f"  Models directory: {args.models_dir}")


if __name__ == "__main__":
    main()
