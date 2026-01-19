# models/hf_causal_neil.py
"""
Simplified vLLM wrapper using pure prefill for log probability computation.

Key insight from Neil: When prefilling a sequence with prompt_logprobs=1, vLLM
returns the exact logprob of each actual token in the sequence. No need for
top-k fallbacks or loading full vocabulary - avoiding OOM issues.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams

    _VLLM_AVAILABLE = True
except ImportError:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    _VLLM_AVAILABLE = False


@dataclass
class GenerateResult:
    texts: List[str]
    gen_token_ids: List[List[int]]
    token_logprobs: Optional[List[List[float]]] = None  # logprobs per token per sample
    total_logprobs: Optional[List[float]] = None  # sum of token logprobs per sample


class HFCausalLMvLLMNew:
    """
    vLLM wrapper that uses pure prefill for exact log probability computation.

    Unlike HFCausalLMvLLM which requests top-k logprobs and searches for tokens,
    this class requests prompt_logprobs=1 which returns the exact logprob of
    the actual token at each position in the prefilled sequence.
    """

    def __init__(
        self,
        model_name: str,
        *,
        tokenizer_name: Optional[str] = None,
        tensor_parallel_size: Optional[int] = None,
        dtype: Optional[str] = None,
        gpu_memory_utilization: float = 0.8,
        **llm_kwargs,
    ) -> None:
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install with `pip install vllm` to use HFCausalLMvLLMNew."
            )

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name, use_fast=True
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        init_kwargs = dict(model=model_name, **llm_kwargs)
        if tensor_parallel_size is not None:
            init_kwargs["tensor_parallel_size"] = tensor_parallel_size
        if dtype is not None:
            init_kwargs["dtype"] = dtype
        if tokenizer_name is not None:
            init_kwargs["tokenizer"] = tokenizer_name
        init_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        # No need for max_logprobs since we only request prompt_logprobs=1

        self.llm = LLM(**init_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ---------- Chat templating ----------
    def build_prompt_text(
        self,
        messages: List[Dict[str, str]],
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        text = ""
        for m in messages:
            role = m.get("role", "user")
            content = m["content"]
            text += f"<<{role}>>: {content}\n"
        if add_generation_prompt:
            text += "<<assistant>>:"
        return text

    # ---------- Generation ----------
    def sample(
        self,
        messages: List[Dict[str, str]],
        n: int = 1,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        enable_thinking: bool = False,
        *,
        return_logprobs: bool = False,
        **generate_kwargs,
    ) -> GenerateResult:
        prompt = self.build_prompt_text(
            messages, enable_thinking=enable_thinking, add_generation_prompt=True
        )
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
        outputs = self.llm.generate([prompt], sampling_params, **generate_kwargs)
        request_output = outputs[0]
        texts: List[str] = []
        gen_token_ids: List[List[int]] = []
        token_logprobs: Optional[List[List[float]]] = [] if return_logprobs else None
        total_logprobs: Optional[List[float]] = [] if return_logprobs else None

        for seq in request_output.outputs:
            texts.append(seq.text)
            gen_token_ids.append(list(seq.token_ids))

            if return_logprobs and seq.logprobs is not None:
                seq_logps: List[float] = []
                for lp_entry in seq.logprobs:
                    if lp_entry:
                        logprob_obj = next(iter(lp_entry.values()))
                        seq_logps.append(float(logprob_obj.logprob))
                    else:
                        seq_logps.append(-1e4)
                token_logprobs.append(seq_logps)
                total_logprobs.append(sum(seq_logps))
            elif return_logprobs and seq.logprobs is None and token_logprobs is not None:
                token_logprobs.append([])
                if total_logprobs is not None:
                    total_logprobs.append(0.0)

        return GenerateResult(
            texts=texts,
            gen_token_ids=gen_token_ids,
            token_logprobs=token_logprobs,
            total_logprobs=total_logprobs,
        )

    # ---------- Teacher-forced log-prob via pure prefill ----------
    def logprob(
        self,
        messages: List[Dict[str, str]],
        completion: str,
        enable_thinking: bool = False,
    ) -> Tuple[float, List[float]]:
        """
        Compute log probability of completion given messages using pure prefill.

        This method prefills the entire (prompt + completion) sequence and extracts
        the logprob of each completion token directly. With prompt_logprobs=1,
        vLLM returns exactly the logprob of the actual token at each position -
        no top-k searching or full-vocab fallback needed.
        """
        results = self.logprob_batch([(messages, completion)], enable_thinking=enable_thinking)
        return results[0]

    def logprob_batch(
        self,
        message_completion_pairs: List[Tuple[List[Dict[str, str]], str]],
        enable_thinking: bool = False,
    ) -> List[Tuple[float, List[float]]]:
        """
        Compute log probabilities for multiple (messages, completion) pairs in batch.

        This is much more efficient than calling logprob() multiple times since
        vLLM can process all sequences in parallel.
        """
        if not message_completion_pairs:
            return []

        # Build all full texts and track prompt lengths
        full_texts: List[str] = []
        prompt_lengths: List[int] = []
        empty_indices: List[int] = []  # Track indices with empty completions

        for idx, (messages, completion) in enumerate(message_completion_pairs):
            prompt_text = self.build_prompt_text(
                messages, enable_thinking=enable_thinking, add_generation_prompt=True
            )
            full_text = prompt_text + completion

            # Tokenize to find where completion starts
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
            comp_ids = full_ids[len(prompt_ids):]

            if not comp_ids:
                empty_indices.append(idx)
                # Still add placeholder to maintain indexing
                full_texts.append(full_text)
                prompt_lengths.append(len(prompt_ids))
            else:
                full_texts.append(full_text)
                prompt_lengths.append(len(prompt_ids))

        # Prefill all sequences in one vLLM call
        sampling_params = SamplingParams(
            n=1,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            prompt_logprobs=1,
            logprobs=0,
        )
        outputs = self.llm.generate(full_texts, sampling_params)

        # Extract results for each sequence
        results: List[Tuple[float, List[float]]] = []
        for idx, (request_output, prompt_len) in enumerate(zip(outputs, prompt_lengths)):
            if idx in empty_indices:
                results.append((0.0, []))
                continue

            prompt_logprobs = request_output.prompt_logprobs
            start_idx = prompt_len
            token_logps: List[float] = []

            for pos in range(start_idx, len(prompt_logprobs)):
                entry = prompt_logprobs[pos]
                if entry is None:
                    token_logps.append(-1e4)
                else:
                    logp = self._extract_single_logprob(entry)
                    token_logps.append(logp if logp is not None else -1e4)

            results.append((sum(token_logps), token_logps))

        return results

    @staticmethod
    def _extract_single_logprob(entry: Any) -> Optional[float]:
        """
        Extract the logprob from a prompt_logprobs entry.

        With prompt_logprobs=1, each entry contains exactly one logprob -
        the logprob of the actual token at that position.
        """
        if entry is None:
            return None

        if isinstance(entry, dict):
            # Dict maps token_id -> Logprob object; take the single entry
            if entry:
                logprob_obj = next(iter(entry.values()))
                return float(logprob_obj.logprob)
            return None

        # Handle object-style entry (older vLLM versions)
        if hasattr(entry, "logprob"):
            return float(entry.logprob)

        return None


