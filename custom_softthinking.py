
import sglang as sgl
import json
import time
from tqdm import tqdm
import argparse
import os
from transformers import AutoTokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
# from matheval import evaluator_map, set_client, AIMEEvaluator
import asyncio
# import matheval
# import humanevaleval
# import mbppeval
from huggingface_hub import HfApi
import torch
import time
import jsonlines
import pickle

import re
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)


def extract_last_boxed(text: str) -> str:
    matches = list(BOXED_RE.finditer(text or ""))
    if matches:
        return matches[-1].group(1).strip()
    return text or ""


def parse_prediction(raw_text: str) -> float:
    """
    Legacy helper used in earlier experiments to map yes/no to a float.
    Kept for compatibility. For boxed numeric accuracy use extract_last_boxed + verify_answer.
    """
    candidate = extract_last_boxed(raw_text)
    candidate = (candidate or raw_text or "").strip().lower()
    if "yes" in candidate and "no" not in candidate:
        return 1.0
    return 0.0

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Process some parameters for text generation.')
    parser.add_argument('--sampling_backend', type=str, choices=["pytorch", "flashinfer"], default="flashinfer", help='Sampling backend')
    parser.add_argument('--model_name', type=str, required=True, default="DeepSeek-R1-Distill-Qwen-1.5B", help='Model name or path')
    parser.add_argument('--num_gpus', type=int, default=8, help='GPU number (tensor parallel size, tp_size)')
    parser.add_argument('--cuda_graph_max_bs', type=int, default=None, help='Max number of batch runned in one time.')
    parser.add_argument('--max_running_requests', type=int, default=None, help='Max number of requests runned together.')
    parser.add_argument('--max_batch', type=int, default=1000000, help='Max number of batch runned in one time.')
    parser.add_argument('--mem_fraction_static', type=float, default=0.5, help='Max memory to use per gpu.')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--output_dir', type=str, default="results", help='Directory to save results')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for processing samples')
    parser.add_argument('--end_idx', type=int, default=500, help='End index for processing samples')

    # sampling parameters
    parser.add_argument('--num_samples', type=int, default=5, help='Sampling number')
    parser.add_argument('--max_generated_tokens', type=int, default=32768, help='Limit the number of generated tokens')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling probability')
    parser.add_argument('--top_k', type=int, default=30, help='Top-k sampling probability')
    parser.add_argument('--min_p', type=float, default=0.0, help='Min-p sampling probability')
    parser.add_argument('--after_thinking_temperature', type=float, default=0.6, help='Temperature after thinking')
    parser.add_argument('--after_thinking_top_p', type=float, default=0.95, help='Top-p after thinking')
    parser.add_argument('--after_thinking_top_k', type=int, default=30, help='Top-k after thinking')
    parser.add_argument('--after_thinking_min_p', type=float, default=0.0, help='Min-p after thinking')
    parser.add_argument('--early_stopping_entropy_threshold', type=float, default=0.0, help='Early stopping entropy threshold (set it to 0.0 to disable early stopping)')
    parser.add_argument('--early_stopping_length_threshold', type=int, default=256, help='Early stopping length threshold')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty')

    # Noise parameters
    parser.add_argument('--dirichlet_alpha', type=float, default=1.0, help='Dirichlet alpha')
    parser.add_argument('--gumbel_softmax_temperature', type=float, default=1.0, help='Gumbel-softmax temperature')
    parser.add_argument('--add_noise_dirichlet', action='store_true', help='Add Dirichlet noise to sampling')
    parser.add_argument('--add_noise_gumbel_softmax', action='store_true', help='Add Gumbel-softmax noise to sampling')

    # Eval & Push parameters
    parser.add_argument('--reeval', action='store_true', help='Enable re-evaluation for code datasets (due to Multiprocessing bug when using sglang rollout)')
    parser.add_argument('--use_llm_judge', action='store_true', help='Enable LLM judge')
    parser.add_argument('--api_base', type=str, default=None, help='')
    parser.add_argument('--deployment_name', type=str, default=None, help='')
    parser.add_argument('--api_version', type=str, default=None, help='')
    parser.add_argument('--api_key', type=str, default=None, help='')
    parser.add_argument('--judge_model_name', type=str, default="gpt-4.1-2025-04-14", help='Judge LLM model name for evaluation')
    parser.add_argument('--push_results_to_hf', action='store_true', help='Enable push to huggingface')
    parser.add_argument('--hf_token', type=str, default=None, help='')
    parser.add_argument('--hf_repo_id', type=str, default=None, help='')

    parser.add_argument("--enable_soft_thinking", action="store_true", help="Enable soft thinking mode")
    parser.add_argument("--think_end_str", type=str, default="</think>")
    parser.add_argument("--max_topk", type=int, default=15)

    args = parser.parse_args()

    model_name = args.model_name
    max_generated_tokens = args.max_generated_tokens
    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    min_p = args.min_p
    think_end_str = args.think_end_str
    num_samples = args.num_samples
    random_seed = args.random_seed
    num_gpus = args.num_gpus
    max_running_requests = args.max_running_requests
    max_batch = args.max_batch
    mem_fraction_static = args.mem_fraction_static
    start_idx = args.start_idx
    end_idx = args.end_idx
    reeval = args.reeval
    # dataset = "ncp"
    dataset = "flawedfictions"
    split = "val"

    nice_name = f"{split}_{temperature}temp_{top_p}topp_{top_k}topk_{min_p}minp_{args.repetition_penalty}reppen_{args.dirichlet_alpha}diralpha_{args.max_topk}maxk_{max_generated_tokens}maxtok_{args.early_stopping_entropy_threshold}enths_{args.early_stopping_length_threshold}lenhs_{args.add_noise_gumbel_softmax}gumbel_{args.add_noise_dirichlet}dirichlet_{args.enable_soft_thinking}softthk_{args.num_samples}nsmpl"


    print(f"Arguments: {args}", flush=True)


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "min_p": min_p, "repetition_penalty": args.repetition_penalty,
                        "after_thinking_temperature": args.after_thinking_temperature, "after_thinking_top_p": args.after_thinking_top_p, "after_thinking_top_k": args.after_thinking_top_k, "after_thinking_min_p": args.after_thinking_min_p,
                        "n": 1, # repeat prompt for num_samples times instead of using num_samples in sampling_params
                        "gumbel_softmax_temperature": args.gumbel_softmax_temperature, "dirichlet_alpha": args.dirichlet_alpha,
                        "max_new_tokens": max_generated_tokens, "think_end_str": think_end_str,
                        "early_stopping_entropy_threshold": args.early_stopping_entropy_threshold,
                        "early_stopping_length_threshold": args.early_stopping_length_threshold
                    }

    os.makedirs(f"{args.output_dir}/results/", exist_ok=True)
    noise_suffix = (
        (f"_gumbel_{args.gumbel_softmax_temperature}" if args.add_noise_gumbel_softmax else "")
        + (f"_dirichlet_{args.dirichlet_alpha}" if args.add_noise_dirichlet else "")
    )
    base_filename = (
        f"{model_name.split('/')[-1]}_{dataset}_{split}_{args.enable_soft_thinking}_{args.num_samples}_"
        f"{temperature}_{top_p}_{top_k}_{min_p}_{args.repetition_penalty}_{args.dirichlet_alpha}_"
        f"{args.max_topk}_{max_generated_tokens}_{args.early_stopping_entropy_threshold}_"
        f"{args.early_stopping_length_threshold}{noise_suffix}"
    )
    results_file = f"{args.output_dir}/results/{dataset}/{base_filename}.json"
    results_statistics_file = f"{args.output_dir}/results/{dataset}/{base_filename}_statistics.json"

    results = []

    print("begin")
    start_time = time.time()

    # # if reeval for code datasets, read results_file
    # if reeval:
    #     # read results_file
    #     with open(results_file, "r") as f:
    #         results = json.load(f)
    #     prompt_list = []
    #     idx_list = list(range(start_idx, min(end_idx,len(results))))
    #     decoded_text_list = []
    #     finish_generation_list = []
    #     generated_tokens_list = []
    #     for r in results:
    #         prompt_list.append(r["prompt"])
    #         decoded_text_list.extend(r["completion"])
    #         finish_generation_list.extend(r["finish_generation"])
    #         generated_tokens_list.extend(r["generated_tokens"])
    #     results = []
    # # if not reeval, collect prompt and idx
    # else:
    #     prompt_list = []
    #     idx_list = []
    #     for idx in range(start_idx, min(end_idx,len(samples))):
    #         sample = samples[idx]

            

    #         prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    #         # Repeat prompt for num_samples times instead of using num_samples in sampling_params
    #         for _ in range(args.num_samples):
    #             prompt_list.append(prompt)

    #         idx_list.append(idx)
    prompt_list = []
    idx_list = []
    answer_list = []
    if dataset == "flawedfictions":
        with jsonlines.open(f"/mnt/disk/latent_tasks/grpo_flawed_fictions/data/{split}.jsonl") as reader:
            for sample in reader:
                prompt = tokenizer.apply_chat_template(sample["prompt"], add_generation_prompt=True, tokenize=False)
                for _ in range(num_samples):
                    prompt_list.append(prompt)
                    idx_list.append(len(prompt_list)-1)
                    answer_list.append(sample["answer"])
    elif dataset == "ncp":
        with jsonlines.open(f"/home/co-guru1/latent/ncp_latent/ncp_latent/rl_data/{split}.jsonl") as reader:
            for sample in reader:
                prompt = tokenizer.apply_chat_template(sample["prompt"], add_generation_prompt=True, tokenize=False)
                for _ in range(num_samples):
                    prompt_list.append(prompt)
                    idx_list.append(len(prompt_list)-1)
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # filter to first 10
    # prompt_list = prompt_list[:4]
    # idx_list = idx_list[:4]

    print(f"len(prompt_list): {len(prompt_list)}")

    # generate results
    decoded_text_list = []
    finish_generation_list = []
    generated_tokens_list = []
    decoded_text_to_prompt = {}
    idx = 0
    while idx < len(prompt_list):
        print(f"Number of GPUs available: {num_gpus}", flush=True)
        llm = sgl.Engine(
            model_path=model_name,
            tp_size=num_gpus,
            log_level="info",
            trust_remote_code=True,
            random_seed=random_seed,
            max_running_requests=max_running_requests,
            mem_fraction_static=mem_fraction_static,
            disable_cuda_graph=True,
            disable_overlap_schedule=True,
            enable_soft_thinking=args.enable_soft_thinking,
            add_noise_dirichlet=args.add_noise_dirichlet,
            add_noise_gumbel_softmax=args.add_noise_gumbel_softmax,
            max_topk=args.max_topk,
            cuda_graph_max_bs=args.cuda_graph_max_bs,
            sampling_backend=args.sampling_backend
        )
        outputs =  llm.generate(prompt_list[idx:idx+max_batch], sampling_params)
        decoded_text_list.extend([o["text"] for o in outputs])
        finish_generation_list.extend([o["meta_info"]["finish_reason"]["type"] == "stop" and not args.enable_soft_thinking for o in outputs])

        generated_tokens_list.extend([o["meta_info"]["completion_tokens"] for o in outputs])
        
        for i, o in enumerate(outputs):
            cur_text = o["text"]
            cur_prompt = prompt_list[idx+i]
            decoded_text_to_prompt[cur_text] = cur_prompt
            # print(f"cur_text: {cur_text}")
            # print(f"cur_prompt: {cur_prompt}")
            # exit()

        idx += max_batch
        outputs = None
        llm.shutdown()

        torch.cuda.empty_cache()

    # evaluate the results
    correct = 0
    for i, decoded_text in enumerate(decoded_text_list):
        answer = answer_list[i]
        predicted = parse_prediction(decoded_text)
        print(f"Predicted: {predicted}, Answer: {answer}", flush=True)
        if int(predicted) == int(answer):
            correct += 1
    print(f"Accuracy: {correct/len(decoded_text_list)}", flush=True)
    # save the resultss
    # with open(f"decoded_text_to_prompt_gumbel_softmax_5n.pkl", "wb") as f:
    # with open(f"decoded_text_to_prompt_gumbel_softmax_5n.pkl", "wb") as f:
    save_name = f"decoded_text_to_prompt_{nice_name}.pkl"
    print(f"Saving to {save_name}", flush=True)
    with open(save_name, "wb") as f:
        pickle.dump(decoded_text_to_prompt, f)

if __name__ == "__main__":
    main()
