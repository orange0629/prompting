import pandas as pd
from tqdm import tqdm
from lib.dataloader import init_benchmark
from lib.modelloader import inference_model
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Script to run predictions with a specified GPU and data file.")
    parser.add_argument("-model_dir", help="Model to evaluate", type=str, default=None, required=True)
    parser.add_argument("-benchmark", help="Benchmark to evaluate", type=str, default="mmlu")
    parser.add_argument("-system_prompts_dir", help="Path to system prompts", type=str, required=True)
    parser.add_argument("-cache_dir", help="Cache location", type=str, default=None)
    parser.add_argument("-multi_thread", help="Multi Thread Inference", type=int, default=1)
    parser.add_argument("-cot", help="Chain-of-Thought Type", type=int, default=0)
    parser.add_argument('--hf', action = 'store_true', help = 'Use Huggingface Transformer', default=False)
    parser.add_argument('-saving_strategy', help="The result types to save", type=str, default="all", choices=['all','eval','raw','none'])
    return parser.parse_args()

args = parse_args()

system_prompts_df = pd.read_csv(args.system_prompts_dir)
system_prompts = system_prompts_df["Prompt"]

model_obj = inference_model(args.model_dir, use_vllm=(not args.hf), cache_dir=args.cache_dir, BATCH_SIZE=16, multi_thread=args.multi_thread)
if args.cot == 1:
    model_obj.model_name += "-CoT"

metric_dict = {}

benchmark_obj = init_benchmark(name=args.benchmark.lower(), cot=args.cot)
q_list = benchmark_obj.load_question_list()
user_prompt = benchmark_obj.get_user_prompt()

for system_prompt in tqdm(system_prompts):
    if system_prompt == "empty":
        system_prompt = ""
    answer_prompts = []
    for q in q_list:
        full_prompt = model_obj.get_prompt_template().format(system_prompt=system_prompt, user_prompt=user_prompt.format(question_prompt=q))
        if benchmark_obj.cot == 1:
            full_prompt += " Let's think step by step. "
        answer_prompts.append(full_prompt)

    outputs = model_obj.generate(answer_prompts, max_token_len=benchmark_obj.get_max_token_len())
    
    metric_dict_single = benchmark_obj.eval_question_list(outputs, save_intermediate=(args.saving_strategy, model_obj.model_name, system_prompt))
    for key in metric_dict_single:
        named_key = f"{model_obj.model_name}/{key}"
        if named_key not in metric_dict:
            metric_dict[named_key] = [metric_dict_single[key]]
        else:
            metric_dict[named_key].append(metric_dict_single[key])

# Read again to prevent overwriting
system_prompts_df = pd.read_csv(args.system_prompts_dir)
for key in metric_dict:
    system_prompts_df[key] = metric_dict[key]
system_prompts_df.to_csv(args.system_prompts_dir, index=False)