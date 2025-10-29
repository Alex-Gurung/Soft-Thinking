#!/usr/bin/env python3
import sglang as sgl
import argparse
import json
import os
import re
import time
from transformers import AutoTokenizer
from typing import List, Dict
import jsonlines
import torch
import pickle
import math
import statistics as stats

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)


def extract_last_boxed(text: str) -> str:
    matches = list(BOXED_RE.finditer(text or ""))
    if matches:
        return matches[-1].group(1).strip()
    return text or ""


def parse_prediction(raw_text: str) -> int:
    candidate = extract_last_boxed(raw_text)
    candidate = (candidate or raw_text or "").strip().lower()
    if "yes" in candidate and "no" not in candidate:
        return 1
    return 0


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
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    if df <= 0:
        return 0.0
    if df in table:
        return table[df]
    return 1.96


def main():
    parser = argparse.ArgumentParser(description='Soft Thinking + richer eval metrics for flawed fictions')
    # Engine / infra
    parser.add_argument('--sampling_backend', type=str, choices=["pytorch", "flashinfer"], default="flashinfer")
    parser.add_argument('--model_name', type=str, required=True, default="./models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--cuda_graph_max_bs', type=int, default=None)
    parser.add_argument('--max_running_requests', type=int, default=None)
    parser.add_argument('--max_batch', type=int, default=1000000)
    parser.add_argument('--mem_fraction_static', type=float, default=0.5)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=1_000_000)

    # Sampling
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--max_generated_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--min_p', type=float, default=0.001)
    parser.add_argument('--after_thinking_temperature', type=float, default=0.6)
    parser.add_argument('--after_thinking_top_p', type=float, default=0.95)
    parser.add_argument('--after_thinking_top_k', type=int, default=30)
    parser.add_argument('--after_thinking_min_p', type=float, default=0.0)
    parser.add_argument('--early_stopping_entropy_threshold', type=float, default=0.01)
    parser.add_argument('--early_stopping_length_threshold', type=int, default=256)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    # Noise
    parser.add_argument('--dirichlet_alpha', type=float, default=1.0)
    parser.add_argument('--gumbel_softmax_temperature', type=float, default=0.5)
    parser.add_argument('--add_noise_dirichlet', action='store_true')
    parser.add_argument('--add_noise_gumbel_softmax', action='store_true')

    # Soft thinking knobs
    parser.add_argument("--enable_soft_thinking", action="store_true")
    parser.add_argument("--think_end_str", type=str, default="</think>")
    parser.add_argument("--max_topk", type=int, default=10)

    args = parser.parse_args()

    model_name = args.model_name
    R = args.num_samples
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    min_p = args.min_p

    # Dataset fixed to flawed fictions val split (matches custom_softthinking)
    dataset = "flawedfictions"
    split = "test"

    nice_name = (
        f"{dataset}_{split}_{temperature}temp_{top_p}topp_{top_k}topk_{min_p}minp_"
        f"{args.repetition_penalty}reppen_{args.dirichlet_alpha}diralpha_"
        f"{args.max_topk}maxk_{args.max_generated_tokens}maxtok_"
        f"{args.early_stopping_entropy_threshold}enths_{args.early_stopping_length_threshold}lenhs_"
        f"{args.add_noise_gumbel_softmax}gumbel_{args.add_noise_dirichlet}dirichlet_"
        f"{args.enable_soft_thinking}softthk_{args.num_samples}nsmpl"
    )

    print(f"Arguments: {args}", flush=True)

    # tokenizer used only for chat template formatting
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "after_thinking_temperature": args.after_thinking_temperature,
        "after_thinking_top_p": args.after_thinking_top_p,
        "after_thinking_top_k": args.after_thinking_top_k,
        "after_thinking_min_p": args.after_thinking_min_p,
        "n": 1,
        "gumbel_softmax_temperature": args.gumbel_softmax_temperature,
        "dirichlet_alpha": args.dirichlet_alpha,
        "max_new_tokens": args.max_generated_tokens,
        "think_end_str": args.think_end_str,
        "early_stopping_entropy_threshold": args.early_stopping_entropy_threshold,
        "early_stopping_length_threshold": args.early_stopping_length_threshold,
    }

    # Build prompts and gold labels
    prompt_list: List[str] = []
    gold_list: List[int] = []
    orig_idx_list: List[int] = []   # original item index for each replicate
    with jsonlines.open(f"/mnt/disk/latent_tasks/grpo_flawed_fictions/data/{split}.jsonl") as reader:
        for idx, sample in enumerate(reader):
            if idx < args.start_idx:
                continue
            if idx >= args.end_idx:
                break
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": sample["prompt"]}], add_generation_prompt=True, tokenize=False)
            gold = int(sample["answer"])  # 0 or 1
            for rep in range(R):
                prompt_list.append(prompt)
                gold_list.append(gold)
                orig_idx_list.append(idx)

    N_items = len(set(orig_idx_list))
    assert len(prompt_list) == len(gold_list)

    print(f"num_items: {N_items}, num_reps: {R}, total_requests: {len(prompt_list)}", flush=True)

    # Run generation in big batches (keep original logic)
    decoded_text_list: List[str] = []
    generated_tokens_list: List[int] = []

    idx = 0
    while idx < len(prompt_list):
        llm = sgl.Engine(
            model_path=model_name,
            tp_size=args.num_gpus,
            log_level="info",
            trust_remote_code=True,
            random_seed=args.random_seed,
            max_running_requests=args.max_running_requests,
            mem_fraction_static=args.mem_fraction_static,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            enable_soft_thinking=args.enable_soft_thinking,
            add_noise_dirichlet=args.add_noise_dirichlet,
            add_noise_gumbel_softmax=args.add_noise_gumbel_softmax,
            max_topk=args.max_topk,
            cuda_graph_max_bs=args.cuda_graph_max_bs,
            sampling_backend=args.sampling_backend,
        )

        batch_prompts = prompt_list[idx: idx + args.max_batch]
        outputs = llm.generate(batch_prompts, sampling_params)

        decoded_text_list.extend([o["text"] for o in outputs])
        generated_tokens_list.extend([o["meta_info"].get("completion_tokens", 0) for o in outputs])

        idx += args.max_batch
        outputs = None
        llm.shutdown()
        torch.cuda.empty_cache()

    assert len(decoded_text_list) == len(gold_list)

    # Evaluate
    preds: List[int] = [parse_prediction(t) for t in decoded_text_list]
    correct_flags: List[int] = [int(p == g) for p, g in zip(preds, gold_list)]

    # Accuracy per repetition: use deterministic layout (grouped by item, then rep)
    acc_per_rep: List[float] = []
    if N_items > 0 and R > 0:
        for rep in range(R):
            # indices i*R + rep for i in 0..N_items-1
            num_ok = 0
            for i in range(N_items):
                idx_flat = i * R + rep
                if idx_flat < len(correct_flags):
                    num_ok += correct_flags[idx_flat]
            acc_per_rep.append(num_ok / N_items)
    else:
        acc_per_rep = [0.0]

    mean_acc = sum(acc_per_rep) / len(acc_per_rep)
    std_acc = stats.stdev(acc_per_rep) if len(acc_per_rep) > 1 else 0.0
    se_acc = (std_acc / math.sqrt(len(acc_per_rep))) if len(acc_per_rep) > 1 else 0.0
    tcrit = t_crit_95(len(acc_per_rep) - 1)
    ci_lo = max(0.0, mean_acc - tcrit * se_acc)
    ci_hi = min(1.0, mean_acc + tcrit * se_acc)

    # Pooled accuracy and Wilson CI
    K = sum(correct_flags)
    T = len(correct_flags)
    pooled_acc = K / T if T else 0.0
    wilson_lo, wilson_hi = wilson_interval(K, T)

    # Per-item instability and expected single-sample accuracy
    disagree_count = 0
    sum_pi = 0.0
    sum_pi_one_minus_pi = 0.0
    for i in range(N_items):
        begin = i * R
        end = begin + R
        item_preds = preds[begin:end]
        item_correct = correct_flags[begin:end]
        pi = sum(item_correct) / R if R else 0.0
        sum_pi += pi
        sum_pi_one_minus_pi += pi * (1 - pi)
        if len(set(item_preds)) > 1:
            disagree_count += 1
    disagree_rate = disagree_count / N_items if N_items else 0.0
    expected_single_sample_acc = sum_pi / N_items if N_items else 0.0
    se_single_sample = math.sqrt(sum_pi_one_minus_pi) / N_items if N_items else 0.0

    # Token stats (completion tokens as reported by the engine)
    if generated_tokens_list:
        tok_mean = stats.mean(generated_tokens_list)
        tok_std = stats.stdev(generated_tokens_list) if len(generated_tokens_list) > 1 else 0.0
        tok_min = min(generated_tokens_list)
        tok_max = max(generated_tokens_list)
        tok_sum = sum(generated_tokens_list)
    else:
        tok_mean = tok_std = tok_min = tok_max = tok_sum = 0

    # Human-friendly report (mirrors tmp_better_eval style)
    print(f"Accuracy per repetition: {[f'{a*100:.2f}%' for a in acc_per_rep]}")
    print(f"Mean accuracy across reps: {mean_acc*100:.2f}%")
    print(f"Std across reps: {std_acc*100:.2f}%")
    if len(acc_per_rep) > 1:
        print(f"95% t-CI across reps: [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]")

    print(f"Pooled accuracy over {T} trials: {pooled_acc*100:.2f}%")
    print(f"Wilson 95% CI (pooled): [{wilson_lo*100:.2f}%, {wilson_hi*100:.2f}%]")
    print(f"Per-item disagreement rate (instability): {disagree_rate*100:.2f}%")
    print(f"Expected accuracy of ONE random sample: {expected_single_sample_acc*100:.2f}% (SE≈{se_single_sample*100:.2f}%)")

    print(
        f"Completion tokens — mean: {tok_mean:.2f}, std: {tok_std:.2f}, min: {tok_min}, max: {tok_max}, total: {tok_sum}"
    )

    # Maintain backward-compatible single-line accuracy for sweep parsers
    print(f"Accuracy: {pooled_acc}", flush=True)

    # Also save prompt mapping for later inspection (as in original)
    decoded_text_to_prompt: Dict[str, str] = {}
    for text, prompt in zip(decoded_text_list, prompt_list):
        decoded_text_to_prompt[text] = prompt
    save_name = f"decoded_text_to_prompt_{nice_name}.pkl"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(save_name, "wb") as f:
        pickle.dump(decoded_text_to_prompt, f)
    print(f"Saved to {save_name}")


if __name__ == "__main__":
    main()

