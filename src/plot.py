"""
Plot IWAE curves from RolloutWithJudge data.

This script takes the output from judge.py (RolloutWithJudge objects in JSONL format)
and computes IWAE curves for each rollout, then creates individual plots and an
aggregate ribbon plot across all rollouts.

Each RolloutWithJudge represents results for a single concept/question with multiple
completions that have been judged. We use the log probabilities and judge scores
to compute importance weights and IWAE bounds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Matplotlib is only used for saving plots to disk.
import matplotlib.pyplot as plt

from .utility import (
    ensure_dir,
    load_yaml,
    parse_args,
    parse_rollouts_with_judge_jsonl,
    RolloutWithJudge,
    save_yaml,
    set_global_seed,
)
from .prbo import iwae_curve, prbo_expectation


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def map_judge_score(judge_score: float) -> float:
    """
    Map judge scores to true scale values.

    Args:
        judge_score: Raw judge score (0, 1, or 2)

    Returns:
        Mapped score value (-100 for 0, -50 for 1, 0 for 2)
    """
    score_mapping = {0: -100.0, 1: -50.0, 2: 0.0, -1: -100.0}
    return score_mapping.get(int(judge_score), 0.0)  # Default to 0 for unknown scores


def resolve_ks(n_samples: int, ks_cfg: List[int] | None) -> List[int]:
    """
    Determine which K values to use for IWAE curve computation.

    Args:
        n_samples: Total number of samples available (max K)
        ks_cfg: Optional list of specific K values from config

    Returns:
        Sorted list of unique K values, filtered to be in range [1, n_samples]
        Default if no config: powers of 2 up to n_samples
    """
    if ks_cfg:
        # Use user-specified K values, filtered to valid range
        ks = [int(k) for k in ks_cfg if int(k) <= n_samples and int(k) >= 1]
        if not ks:
            ks = [1, n_samples]
        return sorted(list(dict.fromkeys(ks)))

    # Default: powers of two up to n_samples
    ks: List[int] = []
    k = 1
    while k < n_samples:
        ks.append(k)
        k *= 2
    ks.append(n_samples)
    return ks


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_iwae_curve(ks: List[int], curve: Dict[int, float], outpath: Path, title: str) -> None:
    """
    Plot a single IWAE curve (for one rollout).

    Args:
        ks: List of K values (number of importance samples)
        curve: Dict mapping K -> IWAE bound value
        outpath: Where to save the plot
        title: Plot title
    """
    xs = np.array(ks, dtype=int)
    ys = np.array([curve.get(int(k), np.nan) for k in xs], dtype=float)

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(xs, ys, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("K (importance samples)")
    plt.ylabel("IWAE bound (nats)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_ribbon(
    ks: List[int],
    per_rollout_curves: List[Dict[int, float]],
    outpath: Path,
    title: str
) -> None:
    """
    Create a ribbon plot showing aggregate IWAE curves across all rollouts.

    Shows mean, median, and 25-75th percentile bands to visualize:
      - Central tendency (mean/median)
      - Variability across rollouts (shaded band)

    Args:
        ks: List of K values
        per_rollout_curves: List of curve dicts, one per rollout
        outpath: Where to save the plot
        title: Plot title
    """
    if not per_rollout_curves:
        return

    # Build matrix: rows = rollouts, cols = K values
    xs = np.array(ks, dtype=int)
    mat = np.array([[c.get(int(k), np.nan) for k in xs] for c in per_rollout_curves], dtype=float)

    # Compute statistics across rollouts for each K
    mean = np.nanmean(mat, axis=0)
    median = np.nanmedian(mat, axis=0)
    p25 = np.nanpercentile(mat, 25, axis=0)
    p75 = np.nanpercentile(mat, 75, axis=0)

    # Plot
    plt.figure(figsize=(7.5, 4.5))
    plt.fill_between(xs, p25, p75, alpha=0.25, label="25–75%")
    plt.plot(xs, median, marker="o", label="median")
    plt.plot(xs, mean, linestyle="--", label="mean")
    plt.xscale("log", base=2)
    plt.xlabel("K (importance samples)")
    plt.ylabel("IWAE bound (nats)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def compute_rollout_iwae(
    rollout: RolloutWithJudge,
    ks: List[int] | None = None,
    resamples: int = 100,
    seed: int | None = None,
) -> Dict[int, float]:
    """
    Compute IWAE curve for a single rollout.

    Args:
        rollout: RolloutWithJudge object with judge scores and log probabilities
        ks: K values to evaluate (if None, uses powers of 2)
        resamples: Number of resamples for IWAE estimation
        seed: Random seed

    Returns:
        Dict mapping K -> IWAE bound value
    """
    # Extract importance weights from the rollout
    logw_list: List[float] = []

    for exp in rollout.completions:
        # Compute importance weight: log w = (log p - log q) + S(y)
        # where S(y) is the mapped judge score (0->-100, 1->-50, 2->0)
        mapped_score = map_judge_score(float(exp.judge_score))
        logw = (exp.p_logp - exp.q_logp) + mapped_score
        logw_list.append(float(logw))

    # Determine K values if not provided
    n_samples = len(logw_list)
    if ks is None:
        ks = resolve_ks(n_samples, None)

    # Filter ks to only include values <= n_samples
    ks = [k for k in ks if k <= n_samples]
    if not ks:
        ks = [n_samples]

    # Compute IWAE curve
    logw_np = np.array(logw_list, dtype=float)
    curve = iwae_curve(
        logw_np,
        ks=ks,
        resamples=resamples,
        seed=seed
    )

    return curve


def run_plot_analysis(cfg: Dict[str, Any]) -> None:
    """
    Run IWAE curve analysis on RolloutWithJudge data.

    Args:
        cfg: Configuration dictionary loaded from YAML
    """
    # Parse config
    input_path = Path(cfg["input_rollouts_jsonl"]).expanduser().resolve()
    output_dir = cfg.get("output_dir", "iwae_analysis")

    scoring_cfg = cfg.get("scoring", {})

    analysis_cfg = cfg.get("analysis", {})
    ks_cfg = analysis_cfg.get("ks", None)
    resamples = int(analysis_cfg.get("resamples", 100))
    seed = analysis_cfg.get("seed", 42)

    out_cfg = cfg.get("output", {})
    make_individual_plots = out_cfg.get("make_individual_plots", True)
    make_ribbon_plot = out_cfg.get("make_ribbon_plot", True)
    save_individual_data = out_cfg.get("save_individual_data", True)
    save_aggregate_data = out_cfg.get("save_aggregate_data", True)

    # Load data
    rollouts = parse_rollouts_with_judge_jsonl(input_path)

    print("=" * 70)
    print(f"IWAE Plot Analysis from Judge Data - {len(rollouts)} rollouts")
    print("=" * 70)

    # Setup output directories
    out_dir = ensure_dir(Path(output_dir))
    plots_dir = ensure_dir(out_dir / "plots")
    indiv_dir = ensure_dir(plots_dir / "individual")
    agg_dir = ensure_dir(plots_dir / "aggregate")
    data_dir = ensure_dir(out_dir / "data")

    # Set seed
    set_global_seed(seed)

    print(f"[info] Analyzing {len(rollouts)} rollouts")
    print(f"[info] Judge score mapping: 0→-100, 1→-50, 2→0")
    print(f"[info] Resamples: {resamples}")

    if not rollouts:
        print("[warn] No rollouts found, exiting.")
        return

    # Initialize result containers
    per_rollout_curves: List[Dict[int, float]] = []
    per_rollout_logw: List[List[float]] = []
    all_ks: List[int] = []  # Will collect all possible K values

    # Process each rollout
    for ri, rollout in enumerate(rollouts):
        print(f"[info] Analyzing rollout {ri+1}/{len(rollouts)} (concept: {rollout.concept})")

        # Compute IWAE curve for this rollout
        curve = compute_rollout_iwae(
            rollout=rollout,
            ks=ks_cfg,
            resamples=resamples,
            seed=seed
        )
        per_rollout_curves.append(curve)

        # Collect importance weights for PRBO computation
        logw_list = []
        for exp in rollout.completions:
            mapped_score = map_judge_score(float(exp.judge_score))
            logw = (exp.p_logp - exp.q_logp) + mapped_score
            logw_list.append(float(logw))
        per_rollout_logw.append(logw_list)

        # Collect K values
        for k in curve.keys():
            if k not in all_ks:
                all_ks.append(k)

        # Plot individual curve if requested
        if make_individual_plots:
            ks_for_plot = sorted(curve.keys())
            out_png = indiv_dir / f"rollout_{ri:04d}_iwae.png"
            title = f"IWAE Curve (Rollout {ri}) - {rollout.concept}"
            plot_iwae_curve(ks_for_plot, curve, out_png, title)

        # Save individual rollout data
        if save_individual_data:
            rollout_data = {
                "rollout_idx": ri,
                "concept": rollout.concept,
                "question": rollout.question,
                "intervention": rollout.intervention.value,
                "qid": rollout.qid,
                "sample_id": rollout.sample_id,
                "num_completions": len(rollout.completions),
                "ks": list(curve.keys()),
                "iwae_curve": curve,
                "importance_weights": logw_list,
            }
            data_path = data_dir / f"rollout_{ri:04d}_data.json"
            with data_path.open("w") as f:
                json.dump(rollout_data, f, indent=2)

    # Sort K values for consistent plotting
    all_ks.sort()

    # Create ribbon plot across all rollouts
    if make_ribbon_plot and per_rollout_curves:
        ribbon_png = agg_dir / "iwae_ribbon_all_rollouts.png"
        title = "IWAE Ribbon Plot - All Rollouts"
        plot_ribbon(all_ks, per_rollout_curves, ribbon_png, title)

    # Save aggregate data
    if save_aggregate_data:
        # Compute PRBO expectations
        prbo_expectations = [
            prbo_expectation(np.array(ws, dtype=float))
            for ws in per_rollout_logw
        ]

        agg_data = {
            "num_rollouts": len(rollouts),
            "total_completions": sum(len(r.completions) for r in rollouts),
            "ks": all_ks,
            "per_rollout_curves": per_rollout_curves,
            "per_rollout_expectations": prbo_expectations,
            "judge_score_mapping": {"0": -100.0, "1": -50.0, "2": 0.0},
            "resamples": resamples,
        }

        # Summary statistics
        if per_rollout_curves:
            # Get IWAE values for each K across all rollouts
            k_stats = {}
            for k in all_ks:
                k_values = []
                for curve in per_rollout_curves:
                    if k in curve:
                        k_values.append(curve[k])

                if k_values:
                    k_stats[str(k)] = {
                        "mean": float(np.mean(k_values)),
                        "median": float(np.median(k_values)),
                        "std": float(np.std(k_values)),
                        "min": float(np.min(k_values)),
                        "max": float(np.max(k_values)),
                        "count": len(k_values),
                    }

            agg_data["k_statistics"] = k_stats

        with (data_dir / "aggregate_analysis.json").open("w") as f:
            json.dump(agg_data, f, indent=2)

    # Save effective config
    effective_cfg = dict(cfg)
    effective_cfg["_resolved"] = {
        "input_path": str(input_path),
        "output_path": str(out_dir),
        "total_rollouts": len(rollouts),
        "total_completions": sum(len(r.completions) for r in rollouts),
    }
    save_yaml(effective_cfg, out_dir / "plot_config.yaml")

    print(f"\n[done] ✓ Analyzed {len(rollouts)} rollouts with {sum(len(r.completions) for r in rollouts)} total completions")
    print(f"[done] ✓ Results saved under: {out_dir}")


def main() -> None:
    args = parse_args(
        description="Plot IWAE curves from RolloutWithJudge data"
    )

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)

    run_plot_analysis(cfg)


if __name__ == "__main__":
    main()

