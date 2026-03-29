"""
Checkpoint Inference
--------------------
Generates model responses at all three checkpoints:
  - Checkpoint 0: Untuned base Phi-3.5 Mini Instruct
  - Checkpoint 1: After Stage 1 (Alpaca fine-tuning)
  - Checkpoint 2: After Stage 2 (Teacher-generated JSON fine-tuning)

For each checkpoint, runs inference on:
  1. The Alpaca held-out eval set
  2. The JSON held-out eval set

Saves all outputs to outputs/<checkpoint_name>/ as JSON files.

Usage:
    # Run all checkpoints
    python inference.py --checkpoints 0 1 2

    # Run only checkpoint 0 (baseline)
    python inference.py --checkpoints 0

    # Custom eval data paths
    python inference.py --checkpoints 0 --alpaca-eval data/alpaca_eval.json --json-eval data/json_eval.json
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# These imports rely on the existing student-model.py module.
# Because the filename uses hyphens, we import via importlib.
# ---------------------------------------------------------------------------
import importlib
student_model = importlib.import_module("student-model")

from config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------
CHECKPOINTS = {
    "0": {
        "name": "checkpoint_0_base",
        "description": "Untuned base model",
        "adapter_path": None,  # no adapter — raw base model
    },
    "1": {
        "name": "checkpoint_1_alpaca",
        "description": "After Stage 1 (Alpaca fine-tuning)",
        "adapter_path": cfg.stage1.output_dir,
    },
    "2": {
        "name": "checkpoint_2_json",
        "description": "After Stage 2 (Teacher JSON fine-tuning)",
        "adapter_path": cfg.stage2.output_dir,
    },
}

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_model_for_checkpoint(checkpoint_id: str):
    """
    Load the appropriate model + tokenizer for a given checkpoint.
    - Checkpoint 0: base model only (no LoRA adapters)
    - Checkpoint 1/2: base model + saved LoRA adapter
    Returns (model, tokenizer).
    """
    ckpt = CHECKPOINTS[checkpoint_id]
    adapter_path = ckpt["adapter_path"]

    tokenizer = student_model.load_tokenizer()

    if adapter_path is None:
        # Checkpoint 0: load the raw quantized base model
        log.info("Loading untuned base model for Checkpoint 0 ...")
        model = student_model.load_base_model()
    else:
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(
                f"Adapter checkpoint not found at '{adapter_path}'. "
                f"Have you completed training for this stage?"
            )
        log.info(f"Loading model with adapter from {adapter_path} ...")
        model, tokenizer = student_model.load_student_from_checkpoint(adapter_path)

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = None,
    temperature: float = None,
    top_p: float = None,
    do_sample: bool = None,
) -> str:
    """Generate a single response using the Phi-3.5 chat template."""
    max_new_tokens = max_new_tokens or cfg.eval.max_new_tokens
    temperature = temperature if temperature is not None else cfg.eval.temperature
    top_p = top_p if top_p is not None else cfg.eval.top_p
    do_sample = do_sample if do_sample is not None else cfg.eval.do_sample

    prompt = student_model.format_phi35_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    output_ids = model.generate(**gen_kwargs)

    # Decode only the newly generated tokens (strip the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    response_ids = output_ids[0][prompt_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return response


def run_inference_on_eval_set(
    model,
    tokenizer,
    eval_data: list[dict],
    eval_name: str,
) -> list[dict]:
    """
    Run inference on an evaluation set and return results.
    Each eval example must have 'instruction' and optionally 'input' and 'output'.
    """
    results = []
    log.info(f"Running inference on {eval_name} ({len(eval_data)} prompts) ...")

    for i, example in enumerate(tqdm(eval_data, desc=eval_name)):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        reference = example.get("output", "")

        start = time.time()
        response = generate_response(model, tokenizer, instruction, input_text)
        elapsed = time.time() - start

        results.append({
            "prompt_id": example.get("prompt_id", f"{eval_name}_{i:04d}"),
            "instruction": instruction,
            "input": input_text,
            "reference": reference,
            "response": response,
            "generation_time_s": round(elapsed, 2),
            "response_token_count": len(tokenizer.encode(response)),
        })

    return results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved {len(results)} results -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run inference at each checkpoint.")
    parser.add_argument(
        "--checkpoints", nargs="+", choices=["0", "1", "2"], default=["0"],
        help="Which checkpoints to run (default: 0)",
    )
    parser.add_argument(
        "--alpaca-eval", default="data/alpaca_eval.json",
        help="Path to Alpaca held-out eval set",
    )
    parser.add_argument(
        "--json-eval", default="data/json_eval.json",
        help="Path to JSON held-out eval set",
    )
    parser.add_argument(
        "--output-dir", default="outputs",
        help="Base directory for saving inference results",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of eval samples (useful for quick testing)",
    )
    args = parser.parse_args()

    # Load eval sets
    eval_sets = {}

    if os.path.exists(args.alpaca_eval):
        with open(args.alpaca_eval) as f:
            alpaca_data = json.load(f)
        if args.max_samples:
            alpaca_data = alpaca_data[:args.max_samples]
        eval_sets["alpaca_eval"] = alpaca_data
        log.info(f"Loaded {len(alpaca_data)} Alpaca eval prompts")
    else:
        log.warning(f"Alpaca eval file not found: {args.alpaca_eval}")

    if os.path.exists(args.json_eval):
        with open(args.json_eval) as f:
            json_data = json.load(f)
        if args.max_samples:
            json_data = json_data[:args.max_samples]
        eval_sets["json_eval"] = json_data
        log.info(f"Loaded {len(json_data)} JSON eval prompts")
    else:
        log.warning(f"JSON eval file not found: {args.json_eval}")

    if not eval_sets:
        log.error("No eval sets found. Run alpaca-data-prep.py first and create json_eval.json.")
        sys.exit(1)

    # Run each checkpoint
    for ckpt_id in args.checkpoints:
        ckpt = CHECKPOINTS[ckpt_id]
        log.info(f"\n{'='*60}")
        log.info(f"CHECKPOINT {ckpt_id}: {ckpt['description']}")
        log.info(f"{'='*60}")

        try:
            model, tokenizer = load_model_for_checkpoint(ckpt_id)
        except FileNotFoundError as e:
            log.warning(f"Skipping checkpoint {ckpt_id}: {e}")
            continue

        for eval_name, eval_data in eval_sets.items():
            results = run_inference_on_eval_set(model, tokenizer, eval_data, eval_name)

            output_path = os.path.join(
                args.output_dir, ckpt["name"], f"{eval_name}_responses.json"
            )
            save_results(results, output_path)

        # Free GPU memory before loading next checkpoint
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info(f"Checkpoint {ckpt_id} complete.\n")

    log.info("All checkpoints done.")


if __name__ == "__main__":
    main()
