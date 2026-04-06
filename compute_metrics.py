"""
compute_metrics.py
Computes ROUGE-L, BERTScore (Alpaca eval) and JSON Validity, Schema Compliance,
Exact Match (JSON eval) for all three checkpoints.

Usage:
    python compute_metrics.py
Outputs results to outputs/eval_results/auto_metrics.json and prints a summary table.
"""

import json
import re
import os
from pathlib import Path

from rouge_score import rouge_scorer
from bert_score import score as bert_score

OUTPUTS = Path("outputs")
RESULTS = OUTPUTS / "eval_results"
CHECKPOINTS = ["checkpoint-0", "checkpoint-1", "checkpoint-2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_code_fence(text: str) -> str:
    """Remove markdown code fences so json.loads can parse the content."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def try_parse_json(text: str):
    """Return parsed JSON or None."""
    try:
        return json.loads(text)
    except Exception:
        try:
            return json.loads(strip_code_fence(text))
        except Exception:
            return None


def normalize_json(obj) -> str:
    """Canonical JSON string for exact-match comparison."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def schema_keys(obj) -> set:
    """Flat set of top-level keys (handles dict and list-of-dicts)."""
    if isinstance(obj, dict):
        return set(obj.keys())
    if isinstance(obj, list):
        keys = set()
        for item in obj:
            if isinstance(item, dict):
                keys.update(item.keys())
        return keys
    return set()


def schema_compliant(response_obj, reference_obj) -> bool:
    """True if response has at least all keys present in reference."""
    ref_keys = schema_keys(reference_obj)
    res_keys = schema_keys(response_obj)
    return ref_keys.issubset(res_keys)


# ---------------------------------------------------------------------------
# Alpaca metrics: ROUGE-L and BERTScore
# ---------------------------------------------------------------------------

def compute_alpaca_metrics(ckpt: str) -> dict:
    path = OUTPUTS / ckpt / "alpaca_eval_responses.json"
    with open(path) as f:
        data = json.load(f)

    hypotheses = [d["response"] for d in data]
    references  = [d["reference"] for d in data]
    n = len(data)

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(ref, hyp)["rougeL"].fmeasure
                    for ref, hyp in zip(references, hypotheses)]
    avg_rouge_l = sum(rouge_scores) / n

    # BERTScore (distilbert is fast and avoids needing a GPU here)
    P, R, F1 = bert_score(
        hypotheses, references,
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
    )
    avg_bertscore_f1 = F1.mean().item()

    # Avg output length
    avg_tokens = sum(d.get("response_token_count", len(d["response"].split()))
                     for d in data) / n

    return {
        "n": n,
        "rouge_l": round(avg_rouge_l, 4),
        "bertscore_f1": round(avg_bertscore_f1, 4),
        "avg_response_tokens": round(avg_tokens, 1),
    }


# ---------------------------------------------------------------------------
# JSON metrics: validity, schema compliance, exact match
# ---------------------------------------------------------------------------

def compute_json_metrics(ckpt: str) -> dict:
    path = OUTPUTS / ckpt / "json_eval_responses.json"
    with open(path) as f:
        data = json.load(f)

    n = len(data)
    valid = 0
    compliant = 0
    exact = 0

    for d in data:
        ref_obj  = try_parse_json(d["reference"])
        resp_obj = try_parse_json(d["response"])

        if resp_obj is not None:
            valid += 1
            if ref_obj is not None and schema_compliant(resp_obj, ref_obj):
                compliant += 1
            if ref_obj is not None and normalize_json(resp_obj) == normalize_json(ref_obj):
                exact += 1

    return {
        "n": n,
        "json_validity":       round(valid     / n, 4),
        "schema_compliance":   round(compliant / n, 4),
        "exact_match":         round(exact     / n, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for ckpt in CHECKPOINTS:
        print(f"\nComputing metrics for {ckpt}...")
        alpaca = compute_alpaca_metrics(ckpt)
        json_m = compute_json_metrics(ckpt)
        all_results[ckpt] = {"alpaca": alpaca, "json": json_m}
        print(f"  Alpaca  — ROUGE-L: {alpaca['rouge_l']:.4f}  BERTScore F1: {alpaca['bertscore_f1']:.4f}  avg tokens: {alpaca['avg_response_tokens']}")
        print(f"  JSON    — Validity: {json_m['json_validity']:.1%}  Schema: {json_m['schema_compliance']:.1%}  Exact: {json_m['exact_match']:.1%}")

    out_path = RESULTS / "auto_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n--- Summary Table ---")
    print(f"{'Checkpoint':<20} {'ROUGE-L':>8} {'BERTScore':>10} {'JSON Valid':>11} {'Schema':>8} {'Exact':>7}")
    print("-" * 68)
    for ckpt, res in all_results.items():
        a, j = res["alpaca"], res["json"]
        label = ckpt.replace("checkpoint-", "Ckpt ")
        print(f"{label:<20} {a['rouge_l']:>8.4f} {a['bertscore_f1']:>10.4f} "
              f"{j['json_validity']:>10.1%} {j['schema_compliance']:>7.1%} {j['exact_match']:>7.1%}")


if __name__ == "__main__":
    main()
