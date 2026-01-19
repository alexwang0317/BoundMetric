"""
Judge script to evaluate completions using a fine-tuned LoRA judge.

Takes RolloutWithUnsteered objects (output of prefill.py) and evaluates each
completion using the judge model, producing RolloutWithJudge objects with
SingleExperimentWithJudge completions containing judge outputs and scores.
"""

from dataclasses import asdict
import json
import re
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .utility import (
    ensure_dir,
    load_yaml,
    parse_args,
    save_yaml,
    set_global_seed,
    RolloutWithJudge,
    RolloutWithUnsteered,
    SingleExperimentWithJudge,
    parse_rollouts_with_unsteered_jsonl,
)


# Judge prompt template from the fine-tuning setup
JUDGE_PROMPT = """Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below.

Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is clearly incorporated. Point out references to the concept in the fragment, noting any minimal or awkward incorporations. After providing your explanation, rate the concept's relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, 1 indicates the concept is somewhat present but minimally or awkwardly incorporated, and 2 indicates the concept is more fully and effectively incorporated, with stronger and more natural integration. YOU MUST provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{response}
[Sentence Fragment End]"""


def extract_rating(text: str) -> int:
    """Extract rating from text in format 'Rating: [[#]]'. Returns -1 if not found."""
    match = re.search(r'Rating:\s*\[\[(\d+)\]\]', text)
    if match:
        return int(match.group(1))
    return -1


def load_judge_model(
    model_path: str,
    gpu_memory_utilization: float = 0.8,
) -> LLM:
    """
    Load judge model with vLLM, following the same process as test.py:
    Load base model + LoRA adapter, merge them, then use with vLLM.

    Args:
        model_path: Path to the LoRA adapter directory (config now specifies adapter path)
        gpu_memory_utilization: GPU memory fraction for vLLM

    Returns:
        vLLM LLM instance with merged LoRA adapter
    """
    print("[info] Loading judge model with LoRA adapter merging...")

    # Base model (same as test.py)
    base_model = "Qwen/Qwen3-8B"
    adapter_path = Path(model_path).expanduser().resolve()

    print(f"[info] Loading base model: {base_model}")
    print(f"[info] Loading adapter from: {adapter_path}")

    # Load the actual base model first (not just the string)
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model_obj, str(adapter_path))

    # Merge the adapter with the base model
    print("[info] Merging LoRA adapter with base model...")
    merged = model.merge_and_unload()

    print("[info] Saving merged model...")
    merged.save_pretrained("merged_model")

    # Clean up memory before loading with vLLM
    del base_model_obj
    del model
    del merged
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("[info] Loading merged model with vLLM...")
    llm = LLM(
        model="merged_model",
        tokenizer=base_model,  # Use the base model name for tokenizer
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,  # Reduce max model length for memory efficiency
    )
    return llm


def judge_rollouts(
    rollouts: List[RolloutWithUnsteered],
    judge_model: LLM,
    sampling_params: SamplingParams,
) -> List[RolloutWithJudge]:
    """
    Judge all completions in rollouts using the judge model.

    Args:
        rollouts: List of RolloutWithUnsteered objects
        judge_model: vLLM LLM instance for the judge
        sampling_params: Sampling parameters for generation

    Returns:
        List of RolloutWithJudge objects with judge outputs and scores
    """
    # Get tokenizer from vLLM model to apply chat template
    tokenizer = judge_model.get_tokenizer()
    
    # Store metadata to map outputs back to rollouts: (rollout_index, completion_index)
    prompt_metadata = [] 
    prompts = []

    print("[info] Building prompts with chat template...")
    for ri, rollout in enumerate(rollouts):
        for ci, exp in enumerate(rollout.completions):
            # Format the raw prompt text
            user_content = JUDGE_PROMPT.format(
                concept=rollout.concept,
                response=exp.completion,
            )

            # Apply Chat Template (Critical for Qwen/Instruction tuned models)
            messages = [{"role": "user", "content": user_content}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            prompts.append(formatted_prompt)
            prompt_metadata.append((ri, ci))

    print(f"[info] Judging {len(prompts)} completions...")

    # Batch generate all judge outputs
    outputs = judge_model.generate(prompts, sampling_params)

    # Verify that we got outputs for all prompts
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs but {len(prompts)} prompts were provided. "
            f"This may indicate a timeout, error, or other issue with the judge model."
        )

    # Map outputs back to rollouts
    # Create a structure to hold results: rollout_idx -> completion_idx -> (output, score)
    results_map: Dict[int, Dict[int, tuple]] = {}
    
    for (ri, ci), output in zip(prompt_metadata, outputs):
        if not output.outputs:
            raise RuntimeError(
                f"vLLM returned empty output for prompt (rollout_idx={ri}, completion_idx={ci})"
            )
        judge_output = output.outputs[0].text.strip()
        judge_score = extract_rating(judge_output)

        if ri not in results_map:
            results_map[ri] = {}
        results_map[ri][ci] = (judge_output, judge_score)

    # Build RolloutWithJudge objects
    judged_rollouts: List[RolloutWithJudge] = []
    for ri, rollout in enumerate(rollouts):
        new_completions: List[SingleExperimentWithJudge] = []

        for ci, exp in enumerate(rollout.completions):
            # Verify that we have a result for this completion
            if ri not in results_map or ci not in results_map[ri]:
                raise RuntimeError(
                    f"Missing judge output for rollout_idx={ri}, completion_idx={ci}. "
                    f"This should not happen if vLLM returned all outputs."
                )
            judge_output, judge_score = results_map[ri][ci]

            new_exp = SingleExperimentWithJudge(
                exp_num=exp.exp_num,
                completion=exp.completion,
                q_logp=exp.q_logp,
                q_token_logprobs=exp.q_token_logprobs,
                p_logp=exp.p_logp,
                p_logprobs=exp.p_logprobs,
                judge_output=judge_output,
                judge_score=judge_score,
            )
            new_completions.append(new_exp)

        new_rollout = RolloutWithJudge(
            intervention=rollout.intervention,
            qid=rollout.qid,
            sample_id=rollout.sample_id,
            concept=rollout.concept,
            question=rollout.question,
            base_prompt=rollout.base_prompt,
            completions=new_completions,
        )
        judged_rollouts.append(new_rollout)

    return judged_rollouts


def save_rollouts_with_judge_jsonl(
    rollouts: List[RolloutWithJudge],
    output_path: Path,
) -> None:
    """Save RolloutWithJudge objects to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for rollout in rollouts:
            record = asdict(rollout)
            # Convert Intervention enum to string for JSON serialization
            record["intervention"] = record["intervention"].value
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_judge(cfg: Dict[str, Any]) -> None:
    """
    Run the judge pipeline to evaluate completions.

    Args:
        cfg: Configuration dictionary with keys:
            - input_rollouts_jsonl: Path to input JSONL with RolloutWithUnsteered data
            - output_dir: Directory to save results
            - judge: Judge model configuration (model_path, gpu_memory_utilization)
            - sampling: Optional sampling config (temperature, max_tokens, seed)
    """
    # Parse config
    input_path = Path(cfg["input_rollouts_jsonl"]).expanduser().resolve()
    out_dir = ensure_dir(Path(cfg.get("output_dir", input_path.parent)))
    output_path = out_dir / "rollouts_with_judge.jsonl"

    judge_cfg = cfg.get("judge", {})
    model_path = judge_cfg.get("model_path")
    if not model_path:
        raise ValueError("judge.model_path must be provided")
    model_path = str(Path(model_path).expanduser().resolve())

    sampling_cfg = cfg.get("sampling", {})
    temperature = float(sampling_cfg.get("temperature", 0.0))
    max_tokens = int(sampling_cfg.get("max_tokens", 512))
    top_p = float(sampling_cfg.get("top_p", 0.95))
    seed = sampling_cfg.get("seed")

    # Set seed if provided
    if seed is not None:
        set_global_seed(seed)

    # Load rollouts
    print(f"[info] Loading rollouts from: {input_path}")
    rollouts = parse_rollouts_with_unsteered_jsonl(input_path)
    print(f"[info] Loaded {len(rollouts)} rollouts")

    if not rollouts:
        print("[warn] No rollouts found, exiting.")
        return

    # Count total completions
    total_completions = sum(len(r.completions) for r in rollouts)
    print(f"[info] Total completions to judge: {total_completions}")

    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for judge model")

    # Load judge model
    judge_model = load_judge_model(
        model_path=model_path,
        gpu_memory_utilization=judge_cfg.get("gpu_memory_utilization", 0.8),
    )

    # Set up sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
    )

    # Judge all rollouts
    print("[info] Running judge on all completions...")
    judged_rollouts = judge_rollouts(rollouts, judge_model, sampling_params)

    # Compute statistics
    all_scores = []
    failed_parses = 0
    for rollout in judged_rollouts:
        for exp in rollout.completions:
            if exp.judge_score == -1:
                failed_parses += 1
            else:
                all_scores.append(exp.judge_score)

    print(f"\n[info] Judge statistics:")
    print(f"  Total completions: {total_completions}")
    print(f"  Successfully parsed: {len(all_scores)}")
    print(f"  Failed to parse: {failed_parses}")
    if all_scores:
        score_dist = {s: all_scores.count(s) for s in sorted(set(all_scores))}
        print(f"  Score distribution: {score_dist}")
        print(f"  Mean score: {sum(all_scores) / len(all_scores):.2f}")

    # Save results
    print(f"\n[info] Saving results to: {output_path}")
    save_rollouts_with_judge_jsonl(judged_rollouts, output_path)

    # Save effective config
    effective_cfg = dict(cfg)
    effective_cfg["_resolved"] = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_rollouts": len(rollouts),
        "total_completions": total_completions,
        "successfully_parsed": len(all_scores),
        "failed_parses": failed_parses,
    }
    save_yaml(effective_cfg, out_dir / "judge_config.yaml")

    print(f"\n[done] ✓ Judged {len(judged_rollouts)} rollouts with {total_completions} completions")
    print(f"[done] ✓ Results saved to: {output_path}")


def main() -> None:
    load_dotenv()

    args = parse_args(
        description="Judge completions using a fine-tuned LoRA judge model"
    )

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)

    run_judge(cfg)


if __name__ == "__main__":
    main()

