import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import pandas as pd
from tqdm import tqdm
from lib.dataloader import init_benchmark
from lib.modelloader import inference_model
import argparse
from filelock import FileLock
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Script to run predictions with a specified GPU and data file.")
    parser.add_argument("-model_dir", help="Model to evaluate", type=str, default=None, required=True)
    parser.add_argument("-benchmark", help="Benchmark to evaluate", type=str, default="mmlu")
    parser.add_argument("-system_prompt", help="Path to system prompts", type=str, required=True)
    parser.add_argument("-task_prompts_ver", help="Task prompt version", type=str, default="base")
    parser.add_argument("-cache_dir", help="Cache location", type=str, default=None)
    parser.add_argument("-multi_thread", help="Multi Thread Inference", type=int, default=1)
    parser.add_argument("-cot", help="Chain-of-Thought Type", type=int, default=0)
    parser.add_argument('--hf', action = 'store_true', help = 'Use Huggingface Transformer', default=False)
    parser.add_argument('-saving_strategy', help="The result types to save", type=str, default="all", choices=['all','eval','raw','none'])
    parser.add_argument('-eval_subset', help="Using a subset of full dataset", type=str, default="all", choices=['all','train','test'])
    return parser.parse_args()

args = parse_args()

with open(f'./data/system_prompts/{args.system_prompt}.md', 'r') as file:
    system_prompt = file.read()

model_name = args.model_dir.split("/")[-1]
model_obj = inference_model(args.model_dir, use_vllm=(not args.hf), cache_dir=args.cache_dir, BATCH_SIZE=16, multi_thread=args.multi_thread)

output_dir = f'''./results/exp_results.jsonl'''

benchmark_lst = args.benchmark.lower().split(",")

for benchmark_name in tqdm(benchmark_lst):
    if benchmark_name.lower().startswith("seacrowd"):
        # Extract dataset name for SEACrowd benchmarks
        dataset_name = benchmark_name.lower().split("_")[1] if "_" in benchmark_name else "khpos"
        benchmark_obj = init_benchmark(name=benchmark_name.lower(), cot=args.cot, seacrowd_params={"dataset_name": dataset_name})
    else:
        benchmark_obj = init_benchmark(name=benchmark_name.lower(), cot=args.cot)

    q_list, eval_range = benchmark_obj.load_random_question_list(num_q=None, split=args.eval_subset)
    if benchmark_name.lower().startswith("seacrowd"):
        user_prompt = "Answer the following question based on the provided dataset:\n{question_prompt}\nAnswer:"
    else:
        with open(f'./data/task_prompts/{benchmark_name}/{args.task_prompts_ver}.md', 'r', encoding="utf-8") as file:
            user_prompt = file.read()

    system_prompt = system_prompt.replace(" /// ", " ")
    answer_prompts = []
    for q in q_list:
        full_prompt = model_obj.get_prompt_template().format(system_prompt=system_prompt, user_prompt=user_prompt.format(question_prompt=q))
        answer_prompts.append(full_prompt)

    outputs = model_obj.generate(answer_prompts, max_token_len=1024)
    
    metric_dict_single = benchmark_obj.eval_question_list(outputs, save_intermediate=(args.saving_strategy, model_obj.model_name, system_prompt), eval_range=eval_range)
    for key in metric_dict_single:
        item_to_write = {"system_prompt": args.system_prompt, "task_prompt": args.task_prompts_ver, "model": model_name, "split": args.eval_subset, "metric": key, "score": metric_dict_single[key]}
        lock_path = output_dir + '.lock'
        lock = FileLock(lock_path)
        with lock:
            with open(output_dir, 'a') as f:
                json.dump(item_to_write, f)
                f.write('\n')