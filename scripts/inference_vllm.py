from vllm import LLM
import pandas as pd
from tqdm import tqdm
from lib.dataloader import init_benchmark
import argparse

cache_dir= "/shared/4/models/"

def parse_args():
    parser = argparse.ArgumentParser(description="Script to run predictions with a specified GPU and data file.")
    parser.add_argument("-model_dir", help="Model to evaluate", type=str, default=None, required=True)
    parser.add_argument("-benchmark", help="Benchmark to evaluate", type=str, default="mmlu")
    parser.add_argument("-system_prompts_dir", help="Path to system prompts", type=str, required=True)
    return parser.parse_args()

args = parse_args()
model_dir = args.model_dir
benchmark = args.benchmark.lower()
system_prompts_dir = args.system_prompts_dir
model_name = model_dir.split("/")[-1] if "/" in model_dir else model_dir


system_prompts_df = pd.read_csv(system_prompts_dir)
system_prompts = system_prompts_df["Prompt"]
llm = LLM(model=model_dir)  # Create an LLM.

# Here's Mingqian's prompt
#template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{{You will be presented with a role-playing context followed by a multiple-choice question. {role_context} Select only the option number that corresponds to the correct answer for the following question.}}\n\n### Input:\n{{{{{question}}} Provide the number of the correct option without explaining your reasoning.}} \n\n### Response:'''
#flan_template = '''{role_context} {question} Please select the correct answer number:'''
#role_context = "You are a helpful assistant."

user_prompt = "The following is a multiple choice question (with answers). Reply with only the option letter.\n{question_prompt}"
llama_template = '''[INST] <<SYS>>
{system_prompt}
<</SYS>>
{user_prompt}[/INST]Answer:'''

metric_dict = {}
benchmark_obj = init_benchmark(name=benchmark)
q_list = benchmark_obj.load_question_list()

for system_prompt in tqdm(system_prompts):
    answer_prompts = []
    for q in q_list:
        full_prompt = llama_template.format(system_prompt=system_prompt, user_prompt=user_prompt.format(question_prompt=q))
        answer_prompts.append(full_prompt)
    outputs = llm.generate(answer_prompts)  # Generate texts from the prompts.
    metric_dict_single = benchmark_obj.eval_question_list(outputs, vllm=True, save_intermediate=(True, f"{model_name}/{system_prompt}"))
    for key in metric_dict_single:
        named_key = f"{model_name}/{key}"
        if named_key not in metric_dict:
            metric_dict[named_key] = [metric_dict_single[key]]
        else:
            metric_dict[named_key].append(metric_dict_single[key])

# Read again to prevent overwriting
system_prompts_df = pd.read_csv(system_prompts_dir)
for key in metric_dict:
    system_prompts_df[key] = metric_dict[key]
system_prompts_df.to_csv(system_prompts_dir, index=False)