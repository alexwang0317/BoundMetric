#!/usr/bin/env python3
"""
Scale sweep script: runs steered rollouts across multiple scale values.
Sweeps scale from 0 to 200 in steps of 10.
"""

import os
import sys
import time
import copy
from pathlib import Path

# Set env vars before vLLM imports
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from dotenv import load_dotenv

from .utility import load_yaml, save_yaml, ensure_dir, parse_args
from .steered import run_single_experiment


def run_scale_sweep(
    base_config_path: Path,
    scale_min: float = 0.0,
    scale_max: float = 200.0,
    scale_step: float = 10.0,
    output_base_dir: str | None = None,
) -> None:
    """
    Run steered rollouts for multiple scale values.
    
    Args:
        base_config_path: Path to the base YAML config
        scale_min: Minimum scale value (default: 0)
        scale_max: Maximum scale value (default: 200)
        scale_step: Step size for scale (default: 10)
        output_base_dir: Base directory for outputs (default: from config)
    """
    # Load base config
    base_cfg = load_yaml(base_config_path)
    if isinstance(base_cfg, list):
        base_cfg = base_cfg[0]
    
    # Get base output directory
    if output_base_dir is None:
        output_base_dir = base_cfg.get("output", {}).get("save_dir", "results/scale_sweep")
    
    # Generate scale values: 0, 10, 20, ..., 200
    scales = []
    current = scale_min
    while current <= scale_max:
        scales.append(current)
        current += scale_step
    
    # Print config summary
    model_name = base_cfg.get("steered_model", {}).get("base_model_name", "unknown")
    vector_path = base_cfg.get("steered_model", {}).get("steer_vector_path", "unknown")
    n_samples = base_cfg.get("sampling", {}).get("iwae_samples", 16)
    questions_path = base_cfg.get("data", {}).get("questions_jsonl", "unknown")
    
    print("\n" + "=" * 80)
    print("                         SCALE SWEEP STARTING")
    print("=" * 80)
    print(f"[config] Model: {model_name}")
    print(f"[config] Vector: {Path(vector_path).name if vector_path else 'N/A'}")
    print(f"[config] Questions: {questions_path}")
    print(f"[config] Samples per question: {n_samples}")
    print("-" * 80)
    print(f"[sweep] Total experiments: {len(scales)}")
    print(f"[sweep] Scale values: {scales}")
    print(f"[sweep] Output base: {output_base_dir}")
    print("=" * 80 + "\n")
    
    sweep_start_time = time.time()
    experiment_times = []  # Track time for each experiment
    completed = 0
    failed = 0
    
    for i, scale in enumerate(scales):
        exp_start_time = time.time()
        
        # Progress header
        progress_pct = (i / len(scales)) * 100
        print("\n" + "=" * 80)
        print(f"[PROGRESS] {i+1}/{len(scales)} ({progress_pct:.0f}%) | Scale = {scale}")
        print("-" * 80)
        
        # ETA calculation
        if experiment_times:
            avg_time = sum(experiment_times) / len(experiment_times)
            remaining = len(scales) - i
            eta_secs = avg_time * remaining
            eta_mins, eta_s = divmod(int(eta_secs), 60)
            eta_hrs, eta_mins = divmod(eta_mins, 60)
            if eta_hrs > 0:
                print(f"[ETA] ~{eta_hrs}h {eta_mins}m remaining ({remaining} experiments left)")
            else:
                print(f"[ETA] ~{eta_mins}m {eta_s}s remaining ({remaining} experiments left)")
        else:
            print(f"[ETA] Calculating after first experiment...")
        
        print("=" * 80)
        
        # Deep copy the config for this run
        cfg = copy.deepcopy(base_cfg)
        
        # Update scale
        cfg["steered_model"]["scale"] = scale
        
        # Update output directory to include scale
        scale_str = f"scale_{scale:.1f}" if scale != int(scale) else f"scale_{int(scale)}"
        out_path = Path(output_base_dir) / scale_str
        cfg["output"]["save_dir"] = str(out_path)
        
        print(f"[info] Output dir: {out_path}")
        print(f"[info] Scale: {scale}")
        print(f"[info] Vector index: {cfg['steered_model'].get('vector_index', 'N/A')}")
        print(f"[info] Target layers: {cfg['steered_model'].get('target_layers', 'default')}")
        print("-" * 40)
        
        # Phase 1: Model initialization
        init_start = time.time()
        print(f"[phase 1/3] Initializing vLLM model...")
        print(f"            Loading: {model_name}")
        print(f"            This may take 30-60 seconds...")
        sys.stdout.flush()
        
        # Run the experiment
        try:
            run_single_experiment(cfg)
            
            exp_elapsed = time.time() - exp_start_time
            experiment_times.append(exp_elapsed)
            completed += 1
            
            exp_mins, exp_secs = divmod(int(exp_elapsed), 60)
            print(f"\n[✓] COMPLETED scale={scale} in {exp_mins}m {exp_secs}s")
            
        except Exception as e:
            exp_elapsed = time.time() - exp_start_time
            failed += 1
            
            print(f"\n[✗] FAILED scale={scale}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Running summary
        total_elapsed = time.time() - sweep_start_time
        t_mins, t_secs = divmod(int(total_elapsed), 60)
        t_hrs, t_mins = divmod(t_mins, 60)
        time_so_far = f"{t_hrs}h {t_mins}m {t_secs}s" if t_hrs > 0 else f"{t_mins}m {t_secs}s"
        
        print(f"[status] Completed: {completed}/{len(scales)} | Failed: {failed} | Time so far: {time_so_far}")
    
    # Final summary
    total_elapsed = time.time() - sweep_start_time
    mins, secs = divmod(int(total_elapsed), 60)
    hours, mins = divmod(mins, 60)
    time_str = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins:02d}:{secs:02d}"
    
    avg_time = sum(experiment_times) / len(experiment_times) if experiment_times else 0
    avg_mins, avg_secs = divmod(int(avg_time), 60)
    
    print("\n" + "=" * 80)
    print("                         SWEEP COMPLETE")
    print("=" * 80)
    print(f"[summary] Total experiments: {len(scales)}")
    print(f"[summary] Completed: {completed}")
    print(f"[summary] Failed: {failed}")
    print(f"[summary] Total time: {time_str}")
    print(f"[summary] Avg time per experiment: {avg_mins}m {avg_secs}s")
    print(f"[summary] Scales tested: {scales}")
    print(f"[summary] Results in: {output_base_dir}")
    print("=" * 80)


def main():
    load_dotenv()
    
    import argparse
    parser = argparse.ArgumentParser(description="Run scale sweep for steered rollouts")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="rl/config/steer.yaml",
        help="Path to base config file",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.0,
        help="Minimum scale value (default: 0)",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=200.0,
        help="Maximum scale value (default: 200)",
    )
    parser.add_argument(
        "--scale-step",
        type=float,
        default=10.0,
        help="Scale step size (default: 10)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: from config)",
    )
    args = parser.parse_args()
    
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    run_scale_sweep(
        base_config_path=config_path,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        scale_step=args.scale_step,
        output_base_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
