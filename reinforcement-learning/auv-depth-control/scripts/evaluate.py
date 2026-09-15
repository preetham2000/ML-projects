#!/usr/bin/env python3
"""Evaluate a saved DDPG or TD3 actor in the AUV simulation."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from auv_control.experiment import (  # noqa: E402
    calculate_metrics,
    evaluate_controller,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actor_path", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps-per-episode", type=int, default=2000)
    parser.add_argument("--start-depth", type=float, default=2.0)
    parser.add_argument("--target-depth", type=float, default=8.0)
    parser.add_argument("--label")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    depths, thetas, ws, qs, actions, rewards = evaluate_controller(
        args.actor_path,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        start_depth=args.start_depth,
        target_depth=args.target_depth,
        plot=args.plot,
        label=args.label,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    metrics = calculate_metrics(
        depths,
        thetas,
        actions,
        rewards,
        args.target_depth,
        dt=0.1,
    )
    for name, values in metrics.items():
        print(f"{name}: {values.tolist()}")


if __name__ == "__main__":
    main()
