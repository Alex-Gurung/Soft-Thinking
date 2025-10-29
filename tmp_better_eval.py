#!/usr/bin/env python3
"""
Evaluate a model served by vLLM (OpenAI-compatible /v1/chat/completions)
on a prepared MuSR test split JSONL (prompt, answer).

Examples:

  # Fine-tuned model on localhost vLLM
  python grpo_musr/eval_testset_vllm.py \
    --model agurung/Qwen2.5-3B-Instruct-musr-grpo \
    --test-file grpo_musr/data/test.jsonl \
    --base-url http://localhost:8000/v1 \
    --max-concurrency 64 --max-tokens 64 \
    --temperature 0.7 --top_p 0.8 --top_k 20

  # Baseline model
  python grpo_musr/eval_testset_vllm.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --test-file grpo_musr/data/test.jsonl \
    --base-url http://localhost:8000/v1 \
    --max-concurrency 64 --max-tokens 64 \
    --temperature 0.7 --top_p 0.8 --top_k 20

Notes:
- Requires a running vLLM server exposing /v1/chat/completions.
- Set OPENAI_API_KEY if your server requires it.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

import math
import statistics as stats
from collections import defaultdict
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)


def extract_last_boxed(text: str) -> str:
    matches = list(BOXED_RE.finditer(text or ""))
    if matches:
        return matches[-1].group(1).strip()
    return text or ""


def parse_prediction(raw_text: str) -> float:
    candidate = extract_last_boxed(raw_text)
    candidate = (candidate or raw_text or "").strip().lower()
    if "yes" in candidate and "no" not in candidate:
        return 1.0
    return 0.0


def chat_completion(
    session: requests.Session,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    # vLLM OpenAI-compatible servers generally accept a "seed" to make sampling reproducible.
    if seed is not None:
        payload["seed"] = seed

    r = session.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@dataclass
class Args:
    model: str
    test_file: str
    base_url: str
    max_concurrency: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    timeout: int
    limit: int
    save_preds: Optional[str]
    num_samples: int
    seed_base: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Evaluate MuSR test split via vLLM chat completions")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--test-file", type=str, default=str(Path(__file__).resolve().parent / "data" / "test.jsonl"))
    p.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--max-concurrency", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save-preds", type=str, default="")
    p.add_argument("--num-samples", type=int, default=10, help="Number of stochastic repetitions per item")
    p.add_argument("--seed-base", type=int, default=0, help="Base seed; actual seed is seed_base + rep")
    a = p.parse_args()
    return Args(
        model=a.model,
        test_file=a.test_file,
        base_url=a.base_url,
        max_concurrency=a.max_concurrency,
        max_tokens=a.max_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        top_k=a.top_k,
        timeout=a.timeout,
        limit=a.limit,
        save_preds=a.save_preds or None,
        num_samples=a.num_samples,
        seed_base=a.seed_base,
    )


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    den = 1 + z * z / n
    center = p_hat + z * z / (2 * n)
    rad = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    lo = max(0.0, (center - rad) / den)
    hi = min(1.0, (center + rad) / den)
    return lo, hi


def t_crit_95(df: int) -> float:
    # Small lookup for two-sided 95% CI; falls back to normal for df>10
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    if df <= 0:
        return 0.0
    if df in table:
        return table[df]
    return 1.96


def main():
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY", "")
    rows = read_jsonl(args.test_file)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    assert rows, f"No rows found in {args.test_file}"

    session = requests.Session()

    tasks = []
    for i, r in enumerate(rows):
        prompt = str(r.get("prompt", ""))
        gold = str(r.get("answer", "")).strip().upper()
        messages = [{"role": "user", "content": prompt}]
        for rep in range(args.num_samples):
            tasks.append((i, rep, gold, messages))

    results: List[Dict[str, Any]] = []

    def _worker(item):
        idx, rep, gold, messages = item
        resp = chat_completion(
            session=session,
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            timeout=args.timeout,
            seed=args.seed_base + rep,
        )
        text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = resp.get("usage", {}) or {}
        pred = parse_prediction(text)
        pred = int(pred)
        gold = int(gold)
        ok = (pred == gold)
        return idx, rep, {
            "idx_original": idx,
            "rep": rep,
            "gold": gold,
            "pred": pred,
            "response": text,
            "correct": ok,
            "num_completion_tokens": usage.get("completion_tokens", 0),
            "num_prompt_tokens": usage.get("prompt_tokens", 0),
        }

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            idx, rep, res = fut.result()
            results.append(res)

    # ---- Metrics ----
    R = args.num_samples
    N = len(rows)
    assert len(results) == N * R, f"Expected {N*R} results, got {len(results)}"

    # Accuracy per repetition
    acc_per_rep = []
    for rep in range(R):
        sub = [r for r in results if r["rep"] == rep]
        acc_per_rep.append(sum(r["correct"] for r in sub) / len(sub))

    mean_acc = sum(acc_per_rep) / R
    std_acc = stats.stdev(acc_per_rep) if R > 1 else 0.0
    se_acc = (std_acc / math.sqrt(R)) if R > 1 else 0.0
    tcrit = t_crit_95(R - 1)
    ci_lo = max(0.0, mean_acc - tcrit * se_acc)
    ci_hi = min(1.0, mean_acc + tcrit * se_acc)

    print(f"Accuracy per repetition: {[f'{a*100:.2f}%' for a in acc_per_rep]}")
    print(f"Mean accuracy across reps: {mean_acc*100:.2f}%")
    print(f"Std across reps: {std_acc*100:.2f}%")
    if R > 1:
        print(f"95% t-CI across reps: [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]")

    # Pooled accuracy over all trials + Wilson CI
    K = sum(r["correct"] for r in results)
    T = len(results)
    pooled_acc = K / T if T else 0.0
    wilson_lo, wilson_hi = wilson_interval(K, T)
    print(f"Pooled accuracy over {T} trials: {pooled_acc*100:.2f}%")
    print(f"Wilson 95% CI (pooled): [{wilson_lo*100:.2f}%, {wilson_hi*100:.2f}%]")

    # Per-item stability diagnostics
    by_item = defaultdict(list)
    for r in results:
        by_item[r["idx_original"]].append(r)

    disagree_count = 0
    sum_pi = 0.0
    sum_pi_one_minus_pi = 0.0
    for i in range(N):
        preds = [r["pred"] for r in by_item[i]]
        corrects = [r["correct"] for r in by_item[i]]
        pi = sum(corrects) / R  # prob correct for that item under your sampling policy
        sum_pi += pi
        sum_pi_one_minus_pi += pi * (1 - pi)
        if len(set(preds)) > 1:
            disagree_count += 1

    disagree_rate = disagree_count / N if N else 0.0
    expected_single_sample_acc = sum_pi / N if N else 0.0
    se_single_sample = math.sqrt(sum_pi_one_minus_pi) / N if N else 0.0

    print(f"Per-item disagreement rate (instability): {disagree_rate*100:.2f}%")
    print(
        f"Expected accuracy of ONE random sample: "
        f"{expected_single_sample_acc*100:.2f}% (SE≈{se_single_sample*100:.2f}%)"
    )

    # ---- Save predictions ----
    def fmt(x: float) -> str:
        s = ("%g" % x)
        return s.replace(".", "p")

    if args.save_preds:
        out_path = Path(args.save_preds)
    else:
        safe_model = args.model.replace("/", "_")
        out_dir = Path("grpo_musr")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (
            f"preds_vllm_{safe_model}_t{fmt(args.temperature)}_tp{fmt(args.top_p)}"
            f"_tk{args.top_k}_R{args.num_samples}_seed{args.seed_base}.jsonl"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            rec = {**r}
            rec["idx"] = i               # running index over all trials
            # Already included: rec["rep"], rec["idx_original"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()

