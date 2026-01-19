from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import argparse
import json
import random
from typing import Any, Dict, List
import numpy as np
import yaml


def ensure_dir(path: Path) -> Path:
    """Create directory and all parent directories if they don't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int | None) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value (None to skip seeding)
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load and parse a YAML configuration file."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as a YAML file."""
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def load_questions(json_path: Path) -> List[str]:
    """
    Load questions from a JSON or JSONL file with flexible format support.

    Supports two formats:

    1. Nested JSON (e.g., data_30.json):
       {
           "DatasetName": {
               "1": {"instruction": "...", ...},
               "2": {"question": "...", ...},
               ...
           },
           ...
       }
       Extracts "instruction", "question", "text", "prompt", or "input" fields.

    2. JSONL (one JSON object per line):
       Each line can be a dict with question fields, a string, or plain text.

    Args:
        json_path: Path to the JSON/JSONL file containing questions

    Returns:
        List of question strings
    """
    questions: List[str] = []

    # First, try to load as a single JSON object (nested format)
    with json_path.open("r") as f:
        content = f.read()

    try:
        data = json.loads(content)

        # Check if it's a nested structure (dict of dicts)
        if isinstance(data, dict) and data:
            first_value = next(iter(data.values()))
            if isinstance(first_value, dict):
                # Nested JSON format - iterate over dataset categories
                for _dataset_name, entries in data.items():
                    if not isinstance(entries, dict):
                        continue

                    # Sort by numeric key to maintain order
                    sorted_keys = sorted(
                        entries.keys(),
                        key=lambda x: int(x) if x.isdigit() else x
                    )

                    for key in sorted_keys:
                        entry = entries[key]
                        if not isinstance(entry, dict):
                            continue

                        # Try common field names for the prompt/question text
                        for field in ("instruction", "question", "text", "prompt", "input"):
                            if field in entry and isinstance(entry[field], str):
                                questions.append(entry[field])
                                break

                if questions:
                    return questions

        # Single dict at top level - try to extract question directly
        if isinstance(data, dict):
            for key in ("question", "text", "prompt", "input", "instruction"):
                if key in data and isinstance(data[key], str):
                    questions.append(data[key])
                    return questions

        # Single string
        if isinstance(data, str):
            return [data]

    except json.JSONDecodeError:
        pass  # Fall through to JSONL parsing

    # Fallback: parse as JSONL (one JSON object per line)
    questions = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, str):
                questions.append(obj)
            elif isinstance(obj, dict):
                # Try common key names for question text
                for key in ("question", "text", "prompt", "input", "instruction"):
                    if key in obj and isinstance(obj[key], str):
                        questions.append(obj[key])
                        break
                else:
                    # Fallback: dump the dict as a string
                    questions.append(json.dumps(obj))
            else:
                questions.append(str(obj))
        except json.JSONDecodeError:
            # Treat line as raw text
            questions.append(line)

    return questions


def resolve_ks(iwae_samples: int, ks_cfg: List[int] | None) -> List[int]:
    """
    Determine which K values to use for IWAE curve computation.

    The IWAE bound is computed for different numbers of importance samples (K).
    This function creates a list of K values to evaluate.

    Args:
        iwae_samples: Total number of samples we're drawing (max K)
        ks_cfg: Optional list of specific K values from config

    Returns:
        Sorted list of unique K values, filtered to be in range [1, iwae_samples]
        Default if no config: powers of 2 up to iwae_samples
    """
    if ks_cfg:
        # Use user-specified K values, filtered to valid range
        ks = [int(k) for k in ks_cfg if int(k) <= iwae_samples and int(k) >= 1]
        if not ks:
            ks = [1, iwae_samples]
        return sorted(list(dict.fromkeys(ks)))

    # Default: powers of two up to iwae_samples
    ks: List[int] = []
    k = 1
    while k < iwae_samples:
        ks.append(k)
        k *= 2
    ks.append(iwae_samples)
    return ks


def convert_to_messages(prompt: str) -> List[Dict[str, str]]:
    """
    Convert a raw string prompt to the message format expected by HFCausalLM.

    HFCausalLM.sample() expects messages in format:
    [{"role": "user", "content": "..."}]

    This helper wraps a plain string prompt into that format.
    """
    return [{"role": "user", "content": prompt}]


def build_steered_prompt(steer_template: str, concept: str) -> str:
    """
    Build the steering prefix by injecting the concept into the template.

    Args:
        steer_template: Template string with {concept} placeholder
        concept: The concept to inject (e.g., "positive sentiment")

    Returns:
        Steered prompt prefix
    """
    try:
        return steer_template.format(concept=concept)
    except Exception:
        # If the template doesn't use .format, just append the concept
        return steer_template + f"\n\n[Concept: {concept}]"


def assemble_generation_prompt(steered_prefix: str, question: str) -> str:
    """
    Assemble the full steered prompt for generation.

    Format: [steered_prefix] + Question: [question] + Answer:

    This creates the prompt we'll use for the steered/proposal distribution (q).
    """
    return f"{steered_prefix.strip()}\n\nQuestion:\n{question.strip()}\n\nAnswer:"


def assemble_base_prompt(question: str) -> str:
    """
    Assemble the base (unsteered) prompt.

    Format: Question: [question] + Answer:

    This creates the prompt for the base distribution (p).
    """
    return f"Question:\n{question.strip()}\n\nAnswer:"


def parse_args(
    description: str | None = None,
    default_config_filename: str = "simply.yaml",
) -> argparse.Namespace:
    """
    Parse command-line arguments for BoundBench scripts.

    Args:
        description: Optional CLI description.
        default_config_filename: Default config file located next to scripts.

    Returns:
        argparse.Namespace with the parsed arguments.
    """
    default_cfg = Path(__file__).parent / default_config_filename
    ap = argparse.ArgumentParser(
        description=description or (
            "Run Propensity Bound-IWAE evaluation with steered prompts and online LLM judging."
        )
    )
    ap.add_argument(
        "-c",
        "--config",
        type=str,
        default=str(default_cfg),
        help="Path to YAML configuration file (default: scripts/simply.yaml)",
    )
    return ap.parse_args()


class Intervention(Enum):
    PROMPT_PREPEND = "prompt_prepend"
    STEERING_VECTORS = "steering_vectors"
    LEREFT = "lsreft"

@dataclass
class SingleExperiment:
    exp_num: int
    completion: str
    q_logp: float
    q_token_logprobs: list[float]

@dataclass
class SingleExperimentWithPrompt(SingleExperiment): 
    p_logp: float
    p_logprobs: list[float]

@dataclass
class SingleExperimentWithJudge(SingleExperimentWithPrompt):
    judge_output: str
    judge_score: int

@dataclass
class Rollout:
    intervention: Intervention
    qid: int
    sample_id: int
    concept: str
    question: str
    base_prompt: str
    completions: list[SingleExperiment]

@dataclass 
class RolloutWithUnsteered:
    intervention: Intervention
    qid: int
    sample_id: int
    concept: str
    question: str
    base_prompt: str
    completions: list[SingleExperimentWithPrompt]

@dataclass
class RolloutWithJudge:
    intervention: Intervention
    qid: int
    sample_id: int
    concept: str
    question: str
    base_prompt: str
    completions: list[SingleExperimentWithJudge]


def parse_rollouts_jsonl(jsonl_path: Path) -> List[Rollout]:
    """
    Parse a JSONL file containing basic rollout data and reconstruct Rollout objects.

    Reads JSONL files containing Rollout data with SingleExperiment completions and reconstructs
    the original Rollout and SingleExperiment dataclass objects.

    Args:
        jsonl_path: Path to the JSONL file to parse

    Returns:
        List of Rollout objects with reconstructed dataclass instances
    """
    import json

    rollouts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            # Convert intervention string back to Intervention enum
            intervention_str = record["intervention"]
            intervention = Intervention(intervention_str)

            # Reconstruct SingleExperiment objects from completions
            completions = []
            for comp_dict in record["completions"]:
                completion = SingleExperiment(
                    exp_num=comp_dict["exp_num"],
                    completion=comp_dict["completion"],
                    q_logp=comp_dict["q_logp"],
                    q_token_logprobs=comp_dict["q_token_logprobs"],
                )
                completions.append(completion)

            # Reconstruct Rollout object
            rollout = Rollout(
                intervention=intervention,
                qid=record["qid"],
                sample_id=record["sample_id"],
                concept=record["concept"],
                question=record["question"],
                base_prompt=record["base_prompt"],
                completions=completions,
            )
            rollouts.append(rollout)

    return rollouts


def parse_rollouts_with_unsteered_jsonl(jsonl_path: Path) -> List[RolloutWithUnsteered]:
    """
    Parse a JSONL file containing rollout data with unsteered completions and reconstruct RolloutWithUnsteered objects.

    Reads JSONL files containing RolloutWithUnsteered data and reconstructs
    the original RolloutWithUnsteered and SingleExperimentWithPrompt dataclass objects.

    Args:
        jsonl_path: Path to the JSONL file to parse

    Returns:
        List of RolloutWithUnsteered objects with reconstructed dataclass instances
    """
    import json

    rollouts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            # Convert intervention string back to Intervention enum
            intervention_str = record["intervention"]
            intervention = Intervention(intervention_str)

            # Reconstruct SingleExperimentWithPrompt objects from completions
            completions = []
            for comp_dict in record["completions"]:
                completion = SingleExperimentWithPrompt(
                    exp_num=comp_dict["exp_num"],
                    completion=comp_dict["completion"],
                    q_logp=comp_dict["q_logp"],
                    q_token_logprobs=comp_dict["q_token_logprobs"],
                    p_logp=comp_dict["p_logp"],
                    p_logprobs=comp_dict["p_logprobs"],
                )
                completions.append(completion)

            # Reconstruct RolloutWithUnsteered object
            rollout = RolloutWithUnsteered(
                intervention=intervention,
                qid=record["qid"],
                sample_id=record["sample_id"],
                concept=record["concept"],
                question=record["question"],
                base_prompt=record["base_prompt"],
                completions=completions,
            )
            rollouts.append(rollout)

    return rollouts


def parse_rollouts_with_judge_jsonl(jsonl_path: Path) -> List[RolloutWithJudge]:
    """
    Parse a JSONL file containing rollout data with judge outputs and reconstruct RolloutWithJudge objects.

    Reads JSONL files containing RolloutWithJudge data and reconstructs
    the original RolloutWithJudge and SingleExperimentWithJudge dataclass objects.

    Args:
        jsonl_path: Path to the JSONL file to parse

    Returns:
        List of RolloutWithJudge objects with reconstructed dataclass instances
    """
    import json

    rollouts = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            # Convert intervention string back to Intervention enum
            intervention_str = record["intervention"]
            intervention = Intervention(intervention_str)

            # Reconstruct SingleExperimentWithJudge objects from completions
            completions = []
            for comp_dict in record["completions"]:
                completion = SingleExperimentWithJudge(
                    exp_num=comp_dict["exp_num"],
                    completion=comp_dict["completion"],
                    q_logp=comp_dict["q_logp"],
                    q_token_logprobs=comp_dict["q_token_logprobs"],
                    p_logp=comp_dict["p_logp"],
                    p_logprobs=comp_dict["p_logprobs"],
                    judge_output=comp_dict["judge_output"],
                    judge_score=comp_dict["judge_score"],
                )
                completions.append(completion)

            # Reconstruct RolloutWithJudge object
            rollout = RolloutWithJudge(
                intervention=intervention,
                qid=record["qid"],
                sample_id=record["sample_id"],
                concept=record["concept"],
                question=record["question"],
                base_prompt=record["base_prompt"],
                completions=completions,
            )
            rollouts.append(rollout)

    return rollouts
