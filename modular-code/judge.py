"""
Judge Evaluation
----------------
Pairwise LLM-as-a-Judge evaluation of student model checkpoints using
Llama 3.1 70B Instruct as the judge.

For each comparison pair (0v1, 1v2, 0v2) and eval type (alpaca, json):
  - Loads inference response files for both checkpoints
  - Presents each prompt to the judge with both responses (randomized A/B order)
  - Collects per-dimension scores + winner declaration as structured JSON
  - Saves results to eval_results/judge_<pair>_<eval_type>.json

Supports resume: skips prompts already present in the output file.

Usage:
    python modular-code/judge.py --pairs 0v1 1v2 0v2 --eval-type both
    python modular-code/judge.py --pairs 1v2 --eval-type alpaca
"""

import argparse
import json
import logging
import os
import random
import re
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_SYSTEM_FILE       = "prompts/judge_system.txt"
DEFAULT_USER_TEMPLATE_FILE = "prompts/judge_user_template.txt"
DEFAULT_OUTPUTS_DIR       = "outputs"
DEFAULT_OUTPUT_DIR        = "eval_results"
CHECKPOINT_INTERVAL       = 50
VALID_PAIRS               = ["0v1", "1v2", "0v2"]

# Maps checkpoint IDs to the directory names used by inference.py
CHECKPOINT_INFO = {
    "0": {"name": "checkpoint_0_base",    "label": "Untuned Base Model"},
    "1": {"name": "checkpoint_1_alpaca",  "label": "After Stage 1 (Alpaca)"},
    "2": {"name": "checkpoint_2_json",    "label": "After Stage 2 (Teacher JSON)"},
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_judge_model(model_id: str):
    """Load Llama 3.1 70B Instruct in 4-bit NF4 quantization."""
    log.info(f"Loading judge model: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        dtype=torch.bfloat16,
    )
    model.eval()
    log.info("Judge model loaded.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_responses(outputs_dir: str, checkpoint_name: str, eval_type: str) -> list[dict]:
    """Load inference responses for a checkpoint + eval type."""
    path = os.path.join(outputs_dir, checkpoint_name, f"{eval_type}_eval_responses.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Response file not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_prompt_files(system_file: str, user_template_file: str) -> tuple[str, str]:
    with open(system_file) as f:
        system_prompt = f.read().strip()
    with open(user_template_file) as f:
        user_template = f.read().strip()
    return system_prompt, user_template


def save_results(results: list, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Judge inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def call_judge(
    model,
    tokenizer,
    system_prompt: str,
    user_message: str,
    max_new_tokens: int = 512,
) -> str:
    """Call the judge model and return the raw text response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = input_ids.shape[1]
    response_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def parse_judge_output(raw: str) -> dict | None:
    """
    Extract structured JSON from judge output.
    Tries: direct parse → code-fence extraction → first {...} block.
    Returns None if all attempts fail.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Pair evaluation
# ---------------------------------------------------------------------------

def run_judge_pair(
    model,
    tokenizer,
    system_prompt: str,
    user_template: str,
    responses_a: list[dict],
    responses_b: list[dict],
    ckpt_id_a: str,
    ckpt_id_b: str,
    eval_type: str,
    output_path: str,
    max_new_tokens: int = 512,
) -> list[dict]:
    """
    Run pairwise judge evaluation for one checkpoint pair and eval type.

    A/B presentation order is randomized per prompt to reduce ordering bias.
    Scores and winner are remapped back to the canonical (checkpoint_a,
    checkpoint_b) labeling before saving.
    """
    ckpt_name_a = CHECKPOINT_INFO[ckpt_id_a]["name"]
    ckpt_name_b = CHECKPOINT_INFO[ckpt_id_b]["name"]

    b_by_id = {r["prompt_id"]: r for r in responses_b}

    # Resume support
    results = []
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_ids = {r["prompt_id"] for r in results}
        log.info(f"Resuming — {len(done_ids)} prompts already done.")

    parse_failures = 0

    for example_a in tqdm(responses_a, desc=f"Judge {ckpt_id_a}v{ckpt_id_b} ({eval_type})"):
        prompt_id = example_a["prompt_id"]
        if prompt_id in done_ids:
            continue
        if prompt_id not in b_by_id:
            log.warning(f"Prompt {prompt_id} missing from checkpoint B — skipping.")
            continue

        example_b = b_by_id[prompt_id]

        # Randomly decide which checkpoint is presented as "A" to the judge
        swap = random.random() < 0.5
        presented_a = example_b if swap else example_a
        presented_b = example_a if swap else example_b

        user_message = user_template.format(
            instruction=presented_a["instruction"],
            input=presented_a.get("input") or "",
            response_a=presented_a["response"],
            response_b=presented_b["response"],
            eval_type=eval_type,
        )

        raw = call_judge(model, tokenizer, system_prompt, user_message, max_new_tokens)
        parsed = parse_judge_output(raw)

        if parsed is None:
            log.warning(f"Parse failure for {prompt_id}. Raw output (200 chars): {raw[:200]}")
            parse_failures += 1
            result = {
                "prompt_id": prompt_id,
                "checkpoint_a": ckpt_name_a,
                "checkpoint_b": ckpt_name_b,
                "parse_error": True,
                "raw_output": raw,
            }
            results.append(result)
            done_ids.add(prompt_id)
            continue

        # Remap scores back to canonical A=checkpoint_a, B=checkpoint_b
        if swap:
            scores_a = parsed.get("response_b_scores")  # judge's B = our A
            scores_b = parsed.get("response_a_scores")  # judge's A = our B
            raw_winner = parsed.get("winner", "")
            winner = {"A": "B", "B": "A"}.get(raw_winner, raw_winner)
        else:
            scores_a = parsed.get("response_a_scores")
            scores_b = parsed.get("response_b_scores")
            winner = parsed.get("winner", "")

        result = {
            "prompt_id": prompt_id,
            "checkpoint_a": ckpt_name_a,
            "checkpoint_b": ckpt_name_b,
            "order_swapped": swap,
            "response_a_scores": scores_a,
            "response_b_scores": scores_b,
            "winner": winner,
            "justification": parsed.get("justification", ""),
            "parse_error": False,
        }
        results.append(result)
        done_ids.add(prompt_id)

        if len(results) % CHECKPOINT_INTERVAL == 0:
            save_results(results, output_path)
            log.info(f"Checkpoint saved: {len(results)} done, {parse_failures} parse failures.")

    save_results(results, output_path)
    log.info(
        f"Pair {ckpt_id_a}v{ckpt_id_b} ({eval_type}) complete — "
        f"{len(results)} results, {parse_failures} parse failures."
    )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge pairwise checkpoint evaluation.")
    parser.add_argument(
        "--pairs", nargs="+", choices=VALID_PAIRS, default=VALID_PAIRS,
        help="Checkpoint pairs to evaluate (default: all three)",
    )
    parser.add_argument(
        "--eval-type", choices=["alpaca", "json", "both"], default="both",
        help="Evaluation suite (default: both)",
    )
    parser.add_argument(
        "--outputs-dir", default=DEFAULT_OUTPUTS_DIR,
        help="Base directory containing checkpoint inference outputs",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to save judge score files",
    )
    parser.add_argument(
        "--system-prompt", default=DEFAULT_SYSTEM_FILE,
        help="Path to judge system prompt file",
    )
    parser.add_argument(
        "--user-template", default=DEFAULT_USER_TEMPLATE_FILE,
        help="Path to judge user message template file",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Max tokens for judge response (default: 512)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for A/B order randomization",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    system_prompt, user_template = load_prompt_files(args.system_prompt, args.user_template)
    eval_types = ["alpaca", "json"] if args.eval_type == "both" else [args.eval_type]

    model, tokenizer = load_judge_model(cfg.judge_model_id)

    for pair in args.pairs:
        ckpt_id_a, ckpt_id_b = pair.split("v")
        for eval_type in eval_types:
            log.info(f"\n{'='*60}")
            log.info(f"Pair: {pair} | Eval: {eval_type}")
            log.info(f"  A: {CHECKPOINT_INFO[ckpt_id_a]['name']}")
            log.info(f"  B: {CHECKPOINT_INFO[ckpt_id_b]['name']}")
            log.info(f"{'='*60}")

            try:
                responses_a = load_responses(args.outputs_dir, CHECKPOINT_INFO[ckpt_id_a]["name"], eval_type)
                responses_b = load_responses(args.outputs_dir, CHECKPOINT_INFO[ckpt_id_b]["name"], eval_type)
            except FileNotFoundError as e:
                log.warning(f"Skipping {pair}/{eval_type}: {e}")
                continue

            output_path = os.path.join(args.output_dir, f"judge_{pair}_{eval_type}.json")
            run_judge_pair(
                model, tokenizer,
                system_prompt, user_template,
                responses_a, responses_b,
                ckpt_id_a, ckpt_id_b,
                eval_type,
                output_path,
                max_new_tokens=args.max_new_tokens,
            )

    log.info("All judge evaluations complete.")


if __name__ == "__main__":
    main()
