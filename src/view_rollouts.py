#!/usr/bin/env python3
"""Simple script to view random rollouts from a rollouts.jsonl file."""

import json
import random
import argparse
from pathlib import Path


def load_rollouts(jsonl_path: Path) -> list:
    """Load rollouts from a JSONL file."""
    rollouts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    return rollouts


def print_rollout(rollout: dict, num_completions: int = 3) -> None:
    """Pretty print a rollout with a few completions."""
    print("=" * 80)
    print(f"Question ID: {rollout['qid']}")
    print(f"Concept: {rollout['concept']}")
    print(f"Intervention: {rollout['intervention']}")
    print("-" * 80)
    print(f"Question:\n{rollout['question'][:500]}...")
    print("-" * 80)
    
    completions = rollout["completions"]
    # Pick random completions to show
    samples = random.sample(completions, min(num_completions, len(completions)))
    
    for i, comp in enumerate(samples):
        print(f"\n[Completion {comp['exp_num']}] (log_prob: {comp['q_logp']:.2f})")
        print("-" * 40)
        # Truncate long completions
        text = comp["completion"]
        if len(text) > 500:
            text = text[:500] + "..."
        print(text)
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="View random rollouts from a JSONL file")
    parser.add_argument(
        "jsonl_path",
        type=str,
        nargs="?",
        default="results/gemma-9b/concept_1/technical_specifications_related_to_audio_products_and_their_performance/rollouts.jsonl",
        help="Path to the rollouts.jsonl file",
    )
    parser.add_argument(
        "-n", "--num-rollouts",
        type=int,
        default=20,
        help="Number of random rollouts to show (default: 20)",
    )
    parser.add_argument(
        "-c", "--completions-per-rollout",
        type=int,
        default=4,
        help="Number of completions to show per rollout (default: 4)",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}")
        return

    rollouts = load_rollouts(jsonl_path)
    
    # Print header with file info
    print("=" * 80)
    print(f"FILE: {jsonl_path.resolve()}")
    print(f"Rollouts: {len(rollouts)} | Completions per rollout: {len(rollouts[0]['completions'])}")
    print("=" * 80 + "\n")

    # Pick random rollouts to display
    sample_rollouts = random.sample(rollouts, min(args.num_rollouts, len(rollouts)))
    
    for rollout in sample_rollouts:
        print_rollout(rollout, num_completions=args.completions_per_rollout)


if __name__ == "__main__":
    main()
