# Must set env vars before vLLM imports
import os
import sys
import yaml
import time

# Note: EasySteer fork uses V1 engine which doesn't support TORCH_SDPA
# We use default attention backend instead
# Disable HuggingFace Xet storage backend (causes network errors)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Fix for RTX 4090 hangs: Disable NCCL P2P and IB communication
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Note: EasySteer V1 engine doesn't support custom attention backends (TORCH_SDPA)
# so we don't set VLLM_ATTENTION_BACKEND here

from .utility import (
    Intervention,
    Rollout,
    SingleExperiment,
    assemble_base_prompt,
    assemble_generation_prompt,
    build_steered_prompt,
    convert_to_messages,
    ensure_dir,
    load_questions,
    load_yaml,
    parse_args,
    save_yaml,
    set_global_seed,
)

from dataclasses import asdict
from vllm.steer_vectors.request import SteerVectorRequest, VectorConfig
import json
from pathlib import Path
from typing import Any, List, Dict, Optional
from vllm import SamplingParams

import torch
from .hf_causal_new import GenerateResult, HFCausalLMvLLMNew
from dotenv import load_dotenv



class HFCausalLMvLLMBatched(HFCausalLMvLLMNew):
    def sample_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        n: int = 1,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        enable_thinking: bool = False,
        return_logprobs: bool = False,
        **generate_kwargs,
    ) -> List[GenerateResult]:
        
        prompts = [
            self.build_prompt_text(
                messages, enable_thinking=enable_thinking, add_generation_prompt=True
            )
            for messages in messages_list
        ]
        
        if not do_sample:
            temperature = 0.0
            top_p = 1.0
            
        sampling_params = SamplingParams(
            n=n,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logprobs=1 if return_logprobs else 0,
        )

        # Call llm.generate with list of prompts
        outputs = self.llm.generate(prompts, sampling_params, **generate_kwargs)
        
        results = []
        for request_output in outputs:
            texts = []
            gen_token_ids = []
            token_logprobs = [] if return_logprobs else None
            total_logprobs = [] if return_logprobs else None
            
            for seq in request_output.outputs:
                texts.append(seq.text)
                gen_token_ids.append(list(seq.token_ids))

                if return_logprobs and seq.logprobs is not None:
                    seq_logps = []
                    for lp_entry in seq.logprobs:
                        if lp_entry:
                            logprob_obj = next(iter(lp_entry.values()))
                            seq_logps.append(float(logprob_obj.logprob))
                        else:
                            seq_logps.append(-1e4)
                    token_logprobs.append(seq_logps)
                    total_logprobs.append(sum(seq_logps))
                elif return_logprobs and seq.logprobs is None:
                     if token_logprobs is not None: token_logprobs.append([])
                     if total_logprobs is not None: total_logprobs.append(0.0)

            results.append(GenerateResult(
                texts=texts,
                gen_token_ids=gen_token_ids,
                token_logprobs=token_logprobs,
                total_logprobs=total_logprobs,
            ))
            
        return results

def _resolve_vector_path(cfg: dict[str, Any]) -> Path:
    """Get steer vector path from config."""
    vector_path = cfg.get("steer_vector_path")
    if not vector_path:
        raise ValueError("You need to provide a steer_vector_path for steering_vectors mode.")
    return Path(vector_path).expanduser().resolve()


def build_steer_request(
    steered_cfg: dict[str, Any],
    request_id: int,
    algorithm: str | None = None,
) -> SteerVectorRequest:
    """Build a SteerVectorRequest from config."""
    vector_path = _resolve_vector_path(steered_cfg)
    if not vector_path.exists():
        raise FileNotFoundError(f"Steer vector not found at {vector_path}")

    if vector_path.suffix == ".pt":
        # Check for index
        idx = steered_cfg.get("vector_index")
        if idx is not None:
             cache_dir = vector_path.parent / ".cache_vecs"
             cache_dir.mkdir(exist_ok=True, parents=True)
             
             cached_name = f"{vector_path.stem}_idx{idx}.pt"
             cached_path = cache_dir / cached_name
             
             if not cached_path.exists():
                 print(f"[info] Extracting vector index {idx} from {vector_path}...")
                 data = torch.load(vector_path, map_location="cpu")
                 
                 print(f"[info] Vector loaded. Shape: {data.shape}")
                 if isinstance(data, torch.Tensor):
                     print(f"[info] Vector L2 Norm: {torch.norm(data.float()).item():.4f}")
                 
                 if isinstance(data, torch.Tensor):
                     if data.dim() > 1:
                         sub = data[int(idx)]
                         torch.save(sub, cached_path)
                     else:
                         torch.save(data, cached_path)
                 else:
                     raise ValueError(f"Expected Tensor in .pt file, got {type(data)}")
             
             vector_path = cached_path

    # Load and print the vector/tensor for debugging
    print(f"[debug] Loading vector from {vector_path} to print first 10 elements...")
    try:
        _tmp_vec = torch.load(vector_path, map_location="cpu")
        if isinstance(_tmp_vec, torch.Tensor):
            print(f"[debug] First 10 elements: {_tmp_vec.flatten()[:10]}")
            print(f"[debug] Norm: {torch.norm(_tmp_vec.float()).item()}")
    except Exception as e:
        print(f"[debug] Could not load vector for printing: {e}")

    # Default to layers 10-25 like test_easysteer.py
    raw_target_layers = steered_cfg.get("target_layers", list(range(10, 26)))
    target_layers = [int(layer) for layer in raw_target_layers]

    prefill_triggers = [int(t) for t in steered_cfg.get("prefill_trigger_tokens", [-1])]
    generate_triggers = [int(t) for t in steered_cfg.get("generate_trigger_tokens", [-1])]

    scale = float(steered_cfg.get("scale", steered_cfg.get("steer_vector_scale", 1.0)))

    request_name = steered_cfg.get("steer_request_name", "steer")

    if algorithm:
        vec_cfg = VectorConfig(
            path=str(vector_path),
            scale=scale,
            target_layers=target_layers,
            prefill_trigger_tokens=prefill_triggers,
            generate_trigger_tokens=generate_triggers,
            algorithm=algorithm,
        )
        return SteerVectorRequest(
            request_name,
            request_id,
            vector_configs=[vec_cfg],
        )

    return SteerVectorRequest(
        request_name,
        request_id,
        steer_vector_local_path=str(vector_path),
        scale=scale,
        target_layers=target_layers,
        prefill_trigger_tokens=prefill_triggers,
        generate_trigger_tokens=generate_triggers,
    )


def run_single_experiment(cfg: dict[str, Any]) -> None: 
    """
    based on the config type, run a single experiment

    Uses HFCausalLMvLLMNew for rollouts first.  
    """ 

    exp_name: str = cfg.get("experiment_name") or "unnamed_experiment"

    steered_cfg = cfg.get("steered_model", {})

    # If steered_cfg.get("mode") is not a valid Intervention value, this will raise a ValueError.
    # If it returns None, or a typo, it will raise as well.
    try:
        raw_mode = Intervention(steered_cfg.get("mode", "prompt_prepend").lower())
    except ValueError as e:
        raise ValueError(f"Invalid steering mode in config: {steered_cfg.get('mode')!r}") from e

    base_model_name: str = steered_cfg["base_model_name"]

    # if the intervention is prompt prepend, we need to get the prompt template
    if raw_mode == Intervention.PROMPT_PREPEND:
        steer_template: str = steered_cfg["prompt_template"]

    data_cfg = cfg.get("data", {})
    concept: str = data_cfg["concept"]
    questions_path = Path(data_cfg["questions_jsonl"]).expanduser().resolve()

    sampling_cfg = cfg.get("sampling", {})
    K: int = int(sampling_cfg.get("iwae_samples", 16))
    max_new_tokens = int(sampling_cfg.get("max_new_tokens", 512))
    temperature = float(sampling_cfg.get("temperature", 0.8))
    top_p = float(sampling_cfg.get("top_p", 0.95))
    enable_thinking = bool(sampling_cfg.get("enable_thinking", False))
    seed = sampling_cfg.get("seed", None)

    # --- Outputs ---
    out_cfg = cfg.get("output", {})
    out_dir = ensure_dir(Path(out_cfg.get("save_dir", f"results/{exp_name}")))
    rollouts_path = out_dir / "rollouts.jsonl"

    # Save the effective configuration for reproducibility
    effective_cfg = dict(cfg)
    effective_cfg["_resolved"] = {
        "iwae_samples": K,
        "questions_path": str(questions_path),
        "output_dir": str(out_dir),
    }
    save_yaml(effective_cfg, out_dir / "exp_config.yaml")

    set_global_seed(seed)

    questions = load_questions(questions_path)
    if not questions:
        raise ValueError(f"No questions loaded from {questions_path}")

    print(f"[info] Loading model: {base_model_name}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HFCausalLMvLLMNew (vLLM-based)")


    # This is for prompt prepend mode 
    if raw_mode == Intervention.PROMPT_PREPEND:
        start_time = time.time()
        steered_prefix = build_steered_prompt(steer_template, concept)

        model = HFCausalLMvLLMNew(
            base_model_name,
            gpu_memory_utilization=steered_cfg.get("gpu_memory_utilization", 0.8),
            enforce_eager=steered_cfg.get("enforce_eager", False),
            tensor_parallel_size=steered_cfg.get("tensor_parallel_size"),
            dtype=steered_cfg.get("dtype"),
        )

        n_total_rollouts = 0

        if out_cfg.get("save_rollouts_jsonl", True):
            rollout_f = rollouts_path.open("w")
        else:
            rollout_f = None

        # --- Main loop (batched generation) ---
        for qi, q in enumerate(questions):
            print(f"\n[info] Processing question {qi+1}/{len(questions)}")

            base_prompt = assemble_base_prompt(q)
            steered_prompt = assemble_generation_prompt(steered_prefix, q)
            steered_messages = convert_to_messages(steered_prompt)

            # Generate K samples at once using vLLM's native batching
            # return_logprobs=True gives us q_token_logprobs from generation
            res: GenerateResult = model.sample(
                steered_messages,
                n=K,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                enable_thinking=enable_thinking,
                return_logprobs=True,
            )

            completions = res.texts if res.texts else []
            q_token_logprobs_list = res.token_logprobs if res.token_logprobs else []
            q_total_logprobs_list = res.total_logprobs if res.total_logprobs else []

            if len(completions) < K:
                # Pad with empty completions if needed
                completions.extend([""] * (K - len(completions)))
                q_token_logprobs_list.extend([[]] * (K - len(q_token_logprobs_list)))
                q_total_logprobs_list.extend([0.0] * (K - len(q_total_logprobs_list)))

            # Build SingleExperiment objects for each sample
            experiment_list: list[SingleExperiment] = []
            for si, completion in enumerate(completions):
                q_logp = q_total_logprobs_list[si] if si < len(q_total_logprobs_list) else 0.0
                q_token_logprobs = q_token_logprobs_list[si] if si < len(q_token_logprobs_list) else []

                exp = SingleExperiment(
                    exp_num=si,
                    completion=completion,
                    q_logp=q_logp,
                    q_token_logprobs=q_token_logprobs,
                )
                experiment_list.append(exp)

            # Create a Rollout for this question with all K completions
            rollout = Rollout(
                intervention=raw_mode,
                qid=qi,
                sample_id=0,  # Single rollout per question
                concept=concept,
                question=q,
                base_prompt=base_prompt,
                completions=experiment_list,
            )

            if rollout_f is not None:
                # Serialize dataclass to dict, convert Intervention enum to string
                record = asdict(rollout)
                record["intervention"] = record["intervention"].value
                rollout_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            n_total_rollouts += len(experiment_list)

        if rollout_f is not None:
            rollout_f.close()

        elapsed_time = time.time() - start_time
        mins, secs = divmod(int(elapsed_time), 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins:02d}:{secs:02d}"
        
        print(f"\n[done] ✓ Completed {n_total_rollouts} rollouts across {len(questions)} questions.")
        print(f"[done] ✓ Total time: {time_str}")
        print(f"[done] ✓ Results saved under: {out_dir}")

    elif raw_mode == Intervention.STEERING_VECTORS:
        start_time = time.time()
        
        # Phase 1: Model initialization
        print(f"\n[phase 1/3] Initializing vLLM + EasySteer model...")
        print(f"            Model: {base_model_name}")
        print(f"            GPU memory utilization: {steered_cfg.get('gpu_memory_utilization', 0.85)}")
        print(f"            Max model len: {steered_cfg.get('max_model_len', 'default')}")
        sys.stdout.flush()
        
        init_start = time.time()
        model = HFCausalLMvLLMBatched(
            base_model_name,
            gpu_memory_utilization=steered_cfg.get("gpu_memory_utilization", 0.85),
            enable_steer_vector=True,
            enforce_eager=steered_cfg.get("enforce_eager", True),
            enable_chunked_prefill=steered_cfg.get("enable_chunked_prefill", False),
            tensor_parallel_size=steered_cfg.get("tensor_parallel_size"),
            dtype=steered_cfg.get("dtype"),
            max_model_len=steered_cfg.get("max_model_len"),
        )
        init_elapsed = time.time() - init_start
        print(f"[phase 1/3] ✓ Model loaded in {init_elapsed:.1f}s")

        n_total_rollouts = 0

        if out_cfg.get("save_rollouts_jsonl", True):
            rollout_f = rollouts_path.open("w")
        else:
            rollout_f = None

        # Phase 2: Prepare requests
        print(f"\n[phase 2/3] Preparing {len(questions)} requests...")
        sys.stdout.flush()
        
        all_messages = []
        all_steer_requests = []

        for qi, q in enumerate(questions):
            base_prompt = assemble_base_prompt(q)
            messages = convert_to_messages(base_prompt)
            all_messages.append(messages)

            steer_request = build_steer_request(
                steered_cfg,
                request_id=steered_cfg.get("steer_request_id_base", 1000) + qi,
                algorithm=steered_cfg.get("algorithm"),
            )
            all_steer_requests.append(steer_request)
        
        print(f"[phase 2/3] ✓ Prepared {len(all_messages)} prompts + steer requests")

        # Phase 3: Generation
        print(f"\n[phase 3/3] Generating {len(questions)} x {K} = {len(questions) * K} completions...")
        print(f"            Max tokens: {max_new_tokens}, Temperature: {temperature}")
        sys.stdout.flush()
        
        gen_start = time.time()
        
        all_results = model.sample_batch(
            all_messages,
            n=K,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            enable_thinking=enable_thinking,
            return_logprobs=True,
            steer_vector_request=all_steer_requests,
        )
        
        gen_elapsed = time.time() - gen_start
        print(f"[phase 3/3] ✓ Generation complete in {gen_elapsed:.1f}s")
        print(f"            ({gen_elapsed/len(questions):.2f}s per question)")

        # Process results
        for i, res in enumerate(all_results):
            qi = i
            q = questions[i]
            base_prompt = assemble_base_prompt(q) # Re-assemble for rollout object
            
            completions = res.texts if res.texts else []
            q_token_logprobs_list = res.token_logprobs if res.token_logprobs else []
            q_total_logprobs_list = res.total_logprobs if res.total_logprobs else []

            # downstream code always gets exactly K elements per input
            if len(completions) < K:
                completions.extend([""] * (K - len(completions)))
                q_token_logprobs_list.extend([[]] * (K - len(q_token_logprobs_list)))
                q_total_logprobs_list.extend([0.0] * (K - len(q_total_logprobs_list)))

            # Build SingleExperiment objects for each sample
            experiment_list: list[SingleExperiment] = []
            for si, completion in enumerate(completions):
                q_logp = q_total_logprobs_list[si]
                q_token_logprobs = q_token_logprobs_list[si]

                exp = SingleExperiment(
                    exp_num=si,
                    completion=completion,
                    q_logp=q_logp,
                    q_token_logprobs=q_token_logprobs,
                )
                experiment_list.append(exp)

            # Create a Rollout for this question with all K completions
            rollout = Rollout(
                intervention=raw_mode,
                qid=qi,
                sample_id=0,
                concept=concept,
                question=q,
                base_prompt=base_prompt,
                completions=experiment_list,
            )

            if rollout_f is not None:
                record = asdict(rollout)
                record["intervention"] = record["intervention"].value
                rollout_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            n_total_rollouts += len(experiment_list)

        if rollout_f is not None:
            rollout_f.close()

        elapsed_time = time.time() - start_time
        mins, secs = divmod(int(elapsed_time), 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins:02d}:{secs:02d}"
        
        print(f"\n[done] ✓ Completed {n_total_rollouts} steered rollouts across {len(questions)} questions.")
        print(f"[done] ✓ Total time: {time_str}")
        print(f"[done] ✓ Results saved under: {out_dir}")

    elif raw_mode == Intervention.LEREFT:
        start_time = time.time()
        # LoReFT-trained vectors: set algorithm="loreft" in VectorConfig
        model = HFCausalLMvLLMNew(
            base_model_name,
            gpu_memory_utilization=steered_cfg.get("gpu_memory_utilization", 0.85),
            enable_steer_vector=True,
            enforce_eager=steered_cfg.get("enforce_eager", True),
            enable_chunked_prefill=steered_cfg.get("enable_chunked_prefill", False),
            tensor_parallel_size=steered_cfg.get("tensor_parallel_size"),
            dtype=steered_cfg.get("dtype"),
        )

        n_total_rollouts = 0

        if out_cfg.get("save_rollouts_jsonl", True):
            rollout_f = rollouts_path.open("w")
        else:
            rollout_f = None

        for qi, q in enumerate(questions):
            print(f"\n[info] Processing question {qi+1}/{len(questions)}")

            base_prompt = assemble_base_prompt(q)
            messages = convert_to_messages(base_prompt)

            steer_request = build_steer_request(
                steered_cfg,
                request_id=steered_cfg.get("steer_request_id_base", 1000) + qi,
                algorithm="loreft",
            )

            res: GenerateResult = model.sample(
                messages,
                n=K,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                enable_thinking=enable_thinking,
                return_logprobs=True,
                steer_vector_request=steer_request,
            )

            completions = res.texts if res.texts else []
            q_token_logprobs_list = res.token_logprobs if res.token_logprobs else []
            q_total_logprobs_list = res.total_logprobs if res.total_logprobs else []

            if len(completions) < K:
                completions.extend([""] * (K - len(completions)))
                q_token_logprobs_list.extend([[]] * (K - len(q_token_logprobs_list)))
                q_total_logprobs_list.extend([0.0] * (K - len(q_total_logprobs_list)))

            experiment_list: list[SingleExperiment] = []
            for si, completion in enumerate(completions):
                q_logp = q_total_logprobs_list[si]
                q_token_logprobs = q_token_logprobs_list[si]

                exp = SingleExperiment(
                    exp_num=si,
                    completion=completion,
                    q_logp=q_logp,
                    q_token_logprobs=q_token_logprobs,
                )
                experiment_list.append(exp)

            rollout = Rollout(
                intervention=raw_mode,
                qid=qi,
                sample_id=0,
                concept=concept,
                question=q,
                base_prompt=base_prompt,
                completions=experiment_list,
            )

            if rollout_f is not None:
                record = asdict(rollout)
                record["intervention"] = record["intervention"].value
                rollout_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            n_total_rollouts += len(experiment_list)

        if rollout_f is not None:
            rollout_f.close()

        elapsed_time = time.time() - start_time
        mins, secs = divmod(int(elapsed_time), 60)
        hours, mins = divmod(mins, 60)
        time_str = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins:02d}:{secs:02d}"
        
        print(f"\n[done] ✓ Completed {n_total_rollouts} LoReFT-steered rollouts across {len(questions)} questions.")
        print(f"[done] ✓ Total time: {time_str}")
        print(f"[done] ✓ Results saved under: {out_dir}")

    else:
        raise ValueError(f"Invalid intervention mode: {raw_mode}")


def main() -> None:
    """
    Entry point for running steered rollout experiments.
    """
    load_dotenv()
    args = parse_args(
        description="Generate steered rollouts with token log probabilities."
    )
    cfg_path = Path(args.config).expanduser().resolve()
    cfg_obj = load_yaml(cfg_path)

    cfg_list = cfg_obj if isinstance(cfg_obj, list) else [cfg_obj]
    for cfg in cfg_list:
        exp_name = cfg.get("experiment_name", "unnamed_experiment")

        # Handle list of concepts
        data_cfg = cfg.get("data", {})
        concepts = data_cfg.get("concept", [])
        if isinstance(concepts, str):
            concepts = [concepts]

        # Capture the base output directory
        out_cfg = cfg.get("output", {})
        base_save_dir = out_cfg.get("save_dir", f"results/{exp_name}")

        for concept in concepts:
            print(f"\n[info] >>> Starting rollout for concept: '{concept}'")

            cfg["data"]["concept"] = concept
            cfg["output"]["save_dir"] = str(Path(base_save_dir) / str(concept))

            run_single_experiment(cfg)

        print(f"[done] ✓ Completed rollouts for: {exp_name}")


if __name__ == "__main__":
    main()


