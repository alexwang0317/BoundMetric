"""
Prefill script to compute unsteered model log probabilities for existing rollouts.

Takes Rollout objects (with SingleExperiment completions) and uses the prefill
technique to compute p(completion | base_prompt) under the unsteered model,
producing RolloutWithUnsteered objects with SingleExperimentWithPrompt completions.
"""

from pathlib import Path

from dataclasses import asdict
import json
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv

from .hf_causal_new import HFCausalLMvLLMNew
from .utility import (
    convert_to_messages,
    ensure_dir,
    load_yaml,
    parse_args,
    save_yaml,
    set_global_seed,
    Rollout,
    RolloutWithUnsteered,
    SingleExperimentWithPrompt,
    parse_rollouts_jsonl,
)


def prefill_logprobs(
    rollouts: List[Rollout],
    model: HFCausalLMvLLMNew,
    enable_thinking: bool = False,
) -> List[RolloutWithUnsteered]:
    """
    Compute unsteered model log probabilities for all completions in rollouts.
    
    For each rollout, uses the base_prompt (without steering) and computes
    p(completion | base_prompt) using the prefill technique.
    
    Args:
        rollouts: List of Rollout objects with SingleExperiment completions
        model: HFCausalLMvLLMNew model instance for computing logprobs
        enable_thinking: Whether to enable thinking mode in chat template
        
    Returns:
        List of RolloutWithUnsteered objects with SingleExperimentWithPrompt completions
    """
    results: List[RolloutWithUnsteered] = []
    
    for rollout in rollouts:
        # Build message format from base_prompt
        messages = convert_to_messages(rollout.base_prompt)
        
        # Collect all (messages, completion) pairs for this rollout
        message_completion_pairs = [
            (messages, exp.completion)
            for exp in rollout.completions
        ]
        
        # Batch compute logprobs for all completions
        logprob_results = model.logprob_batch(
            message_completion_pairs,
            enable_thinking=enable_thinking,
        )
        
        # Convert SingleExperiment -> SingleExperimentWithPrompt
        new_completions: List[SingleExperimentWithPrompt] = []
        for exp, (p_logp, p_logprobs) in zip(rollout.completions, logprob_results):
            new_exp = SingleExperimentWithPrompt(
                exp_num=exp.exp_num,
                completion=exp.completion,
                q_logp=exp.q_logp,
                q_token_logprobs=exp.q_token_logprobs,
                p_logp=p_logp,
                p_logprobs=p_logprobs,
            )
            new_completions.append(new_exp)
        
        # Create RolloutWithUnsteered
        new_rollout = RolloutWithUnsteered(
            intervention=rollout.intervention,
            qid=rollout.qid,
            sample_id=rollout.sample_id,
            concept=rollout.concept,
            question=rollout.question,
            base_prompt=rollout.base_prompt,
            completions=new_completions,
        )
        results.append(new_rollout)
    
    return results


def save_rollouts_with_unsteered_jsonl(
    rollouts: List[RolloutWithUnsteered],
    output_path: Path,
) -> None:
    """Save RolloutWithUnsteered objects to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for rollout in rollouts:
            record = asdict(rollout)
            # Convert Intervention enum to string for JSON serialization
            record["intervention"] = record["intervention"].value
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_prefill(cfg: Dict[str, Any]) -> None:
    """
    Run the prefill pipeline to compute unsteered logprobs.
    
    Args:
        cfg: Configuration dictionary with keys:
            - input_rollouts_jsonl: Path to input JSONL with Rollout data
            - output_dir: Directory to save results
            - model: Model configuration (base_model_name, etc.)
            - sampling: Optional sampling config (enable_thinking, seed)
    """
    # Parse config
    input_path = Path(cfg["input_rollouts_jsonl"]).expanduser().resolve()
    out_dir = ensure_dir(Path(cfg.get("output_dir", input_path.parent)))
    output_path = out_dir / "rollouts_with_unsteered.jsonl"
    
    model_cfg = cfg.get("model", {})
    base_model_name = model_cfg.get("base_model_name")
    if not base_model_name:
        raise ValueError("model.base_model_name must be provided")
    
    sampling_cfg = cfg.get("sampling", {})
    enable_thinking = bool(sampling_cfg.get("enable_thinking", False))
    seed = sampling_cfg.get("seed")
    
    # Set seed if provided
    if seed is not None:
        set_global_seed(seed)
    
    # Load rollouts
    print(f"[info] Loading rollouts from: {input_path}")
    rollouts = parse_rollouts_jsonl(input_path)
    print(f"[info] Loaded {len(rollouts)} rollouts")
    
    if not rollouts:
        print("[warn] No rollouts found, exiting.")
        return
    
    # Count total completions
    total_completions = sum(len(r.completions) for r in rollouts)
    print(f"[info] Total completions to process: {total_completions}")
    
    # Initialize model
    print(f"[info] Loading model: {base_model_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prefill logprob computation")
    
    model = HFCausalLMvLLMNew(
        base_model_name,
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.8),
        tensor_parallel_size=model_cfg.get("tensor_parallel_size"),
        dtype=model_cfg.get("dtype"),
    )
    
    # Compute unsteered logprobs
    print("[info] Computing unsteered logprobs via prefill...")
    results = prefill_logprobs(rollouts, model, enable_thinking=enable_thinking)
    
    # Save results
    print(f"[info] Saving results to: {output_path}")
    save_rollouts_with_unsteered_jsonl(results, output_path)
    
    # Save effective config
    effective_cfg = dict(cfg)
    effective_cfg["_resolved"] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_rollouts": len(rollouts),
        "total_completions": total_completions,
    }
    save_yaml(effective_cfg, out_dir / "prefill_config.yaml")
    
    print(f"\n[done] ✓ Processed {len(results)} rollouts with {total_completions} completions")
    print(f"[done] ✓ Results saved to: {output_path}")


def main() -> None:
    load_dotenv()
    
    args = parse_args(
        description="Use prefill to get unsteered model log probabilities for existing rollouts"
    )
    
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)
    
    run_prefill(cfg)


if __name__ == "__main__":
    main()


