#!/usr/bin/env python3
"""Train one of the project's hand-written RL agents."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from auv_control.experiment import train_agent  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=("ddpg", "td3"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps-per-episode", type=int, default=200)
    parser.add_argument("--start-depth", type=float, default=2.0)
    parser.add_argument("--target-depth", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train_agent(
        algorithm=args.algorithm,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        start_depth=args.start_depth,
        target_depth=args.target_depth,
        output_dir=args.output_dir,
        plot=args.plot,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
