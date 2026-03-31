"""
Teacher-Generated JSON Instruct Dataset
-----------------------------------------
Imitation learning pipeline: feeds task prompts to Llama 3.1 70B Instruct
and collects validated JSON responses as training data for the student model.

Steps:
  1. Load prompt bank from prompts/teacher_gen_prompts.json (editable)
  2. Load system prompt from prompts/teacher_gen_system.txt (editable)
  3. Run each prompt through the teacher model
  4. Validate every response as JSON; retry up to --max-retries on failure
  5. Save validated examples to data/teacher_generated.json

Usage:
    python modular-code/teacher-gen-data.py \\
        --output-path data/teacher_generated.json \\
        --num-samples 5000 \\
        --validate-json
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

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

# Default paths (relative to project root)
DEFAULT_PROMPTS_FILE = "prompts/teacher_gen_prompts.json"
DEFAULT_SYSTEM_FILE  = "prompts/teacher_gen_system.txt"
DEFAULT_OUTPUT_PATH  = "data/teacher_generated.json"
CHECKPOINT_INTERVAL  = 50   # save intermediate results every N successful examples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_teacher_model(model_id: str):
    """Load Llama 3.1 70B Instruct in 4-bit NF4 quantization for inference."""
    log.info(f"Loading teacher model: {model_id}")
    log.info("Using 4-bit NF4 quantization (~35-40 GB VRAM required)")

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
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    log.info("Teacher model loaded.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, system_prompt: str, instruction: str, input_text: str) -> str:
    """Format the request using Llama 3.1's chat template via the tokenizer."""
    user_content = instruction if not input_text.strip() else f"{instruction}\n\n{input_text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Run a single forward pass and return the decoded completion."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# JSON extraction and validation
# ---------------------------------------------------------------------------

def extract_and_validate_json(text: str) -> tuple[str | None, bool]:
    """
    Try to parse valid JSON from the raw model output.
    Returns (json_string, is_valid).

    Handles:
      - Clean JSON output (ideal case)
      - JSON wrapped in ```json ... ``` markdown fences
      - JSON preceded/followed by explanatory text
    """
    # 1. Direct parse
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False), True
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            return json.dumps(obj, ensure_ascii=False), True
        except json.JSONDecodeError:
            pass

    # 3. Greedily extract the outermost { } or [ ]
    for pattern in (r"(\{[\s\S]+\})", r"(\[[\s\S]+\])"):
        match = re.search(pattern, text)
        if match:
            try:
                obj = json.loads(match.group(1))
                return json.dumps(obj, ensure_ascii=False), True
            except json.JSONDecodeError:
                pass

    return None, False


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def run_generation(
    model,
    tokenizer,
    prompt_bank: list[dict],
    system_prompt: str,
    num_samples: int,
    max_retries: int,
    output_path: str,
) -> list[dict]:
    """
    Generate teacher responses for the prompt bank.

    If num_samples > len(prompt_bank), the bank is cycled (shuffled each pass)
    to fill the quota.  For maximum diversity, expand teacher_gen_prompts.json.
    """
    # Build the list of prompts to process
    if num_samples <= len(prompt_bank):
        prompts = random.sample(prompt_bank, num_samples)
    else:
        log.warning(
            f"num_samples ({num_samples}) > prompt bank size ({len(prompt_bank)}). "
            "Cycling through bank with shuffling. Add more prompts to "
            "prompts/teacher_gen_prompts.json for greater diversity."
        )
        import math
        repeats = math.ceil(num_samples / len(prompt_bank))
        extended = []
        for _ in range(repeats):
            shuffled = prompt_bank.copy()
            random.shuffle(shuffled)
            extended.extend(shuffled)
        prompts = extended[:num_samples]

    results: list[dict] = []
    skipped = 0

    for i, prompt_entry in enumerate(tqdm(prompts, desc="Generating")):
        instruction = prompt_entry["instruction"]
        input_text  = prompt_entry.get("input", "")
        task_type   = prompt_entry.get("task_type", "unknown")

        full_prompt = build_prompt(tokenizer, system_prompt, instruction, input_text)
        success = False

        for attempt in range(1, max_retries + 1):
            raw = generate_response(model, tokenizer, full_prompt)
            clean_json, valid = extract_and_validate_json(raw)

            if valid:
                results.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": clean_json,
                    "task_type": task_type,
                })
                success = True
                break
            else:
                log.debug(
                    f"[{i+1}/{len(prompts)}] Attempt {attempt}/{max_retries} "
                    f"invalid JSON for task_type={task_type}"
                )

        if not success:
            log.warning(
                f"[{i+1}/{len(prompts)}] Skipped after {max_retries} failed attempts "
                f"(task_type={task_type})"
            )
            skipped += 1

        # Checkpoint save
        if len(results) > 0 and len(results) % CHECKPOINT_INTERVAL == 0:
            _save(results, output_path)
            log.info(f"Checkpoint: {len(results)} examples saved.")

    log.info(
        f"Generation complete: {len(results)} valid examples, "
        f"{skipped} skipped out of {len(prompts)} total."
    )
    return results


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(results: list[dict]) -> None:
    from collections import Counter
    counts = Counter(r["task_type"] for r in results)
    log.info("--- Dataset statistics ---")
    log.info(f"  Total examples : {len(results)}")
    for task_type, count in sorted(counts.items()):
        log.info(f"  {task_type:<40} {count}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate teacher JSON instruct dataset")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH,
                        help="Path to save generated dataset JSON")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Max examples to generate (default: use full prompt bank)")
    parser.add_argument("--prompts-file", default=DEFAULT_PROMPTS_FILE,
                        help="Path to teacher_gen_prompts.json (editable prompt bank)")
    parser.add_argument("--system-file", default=DEFAULT_SYSTEM_FILE,
                        help="Path to teacher_gen_system.txt (system prompt)")
    parser.add_argument("--model-id", default=cfg.teacher_model_id,
                        help="HuggingFace model ID for the teacher")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max regeneration attempts per prompt on invalid JSON")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max new tokens per teacher response")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-json", action="store_true",
                        help="(Flag kept for SLURM compatibility — validation is always on)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load prompt bank
    if not os.path.exists(args.prompts_file):
        raise FileNotFoundError(
            f"Prompt bank not found: '{args.prompts_file}'. "
            "Expected prompts/teacher_gen_prompts.json."
        )
    with open(args.prompts_file, encoding="utf-8") as f:
        prompt_bank: list[dict] = json.load(f)
    log.info(f"Prompt bank loaded: {len(prompt_bank)} prompts from '{args.prompts_file}'")

    # Load system prompt
    if not os.path.exists(args.system_file):
        raise FileNotFoundError(
            f"System prompt not found: '{args.system_file}'. "
            "Expected prompts/teacher_gen_system.txt."
        )
    with open(args.system_file, encoding="utf-8") as f:
        system_prompt = f.read().strip()
    log.info(f"System prompt loaded from '{args.system_file}'")

    num_samples = args.num_samples or len(prompt_bank)

    # Load teacher model
    model, tokenizer = load_teacher_model(args.model_id)

    # Generate
    os.makedirs("data", exist_ok=True)
    results = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_bank=prompt_bank,
        system_prompt=system_prompt,
        num_samples=num_samples,
        max_retries=args.max_retries,
        output_path=args.output_path,
    )

    # Final save
    _save(results, args.output_path)
    log.info(f"Saved {len(results)} examples to '{args.output_path}'")
    print_stats(results)
