from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
from lib.dataloader import init_benchmark
from lib.hfmodel import hfmodel
import argparse

llama2_template = '''[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt}[/INST]Answer:'''
llama3_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer:'''
mixtral_template = '''<s> [INST] {system_prompt}\n{user_prompt} [/INST] Answer:'''
dbrx_template = '''<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\nAnswer:'''
jamba_template = '''<|startoftext|>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST] Answer:'''
qwen_template = '''<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\nAnswer:'''
gemma_template = '''<bos><start_of_turn>user\n{system_prompt}\n{user_prompt}<end_of_turn>\n<start_of_turn>model\nAnswer:'''
commandR_template = '''<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Answer:'''
llm_template_dict = {"llama-2": llama2_template, "llama-3": llama3_template, "mixtral": mixtral_template, "dbrx": dbrx_template, "jamba": jamba_template, "qwen": qwen_template, "gemma": gemma_template, "command-r": commandR_template}

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
model_dir = args.model_dir
cache_dir = args.cache_dir
benchmark = args.benchmark.lower()
system_prompts_dir = args.system_prompts_dir
model_name = model_dir.split("/")[-1] if "/" in model_dir else model_dir
if args.cot == 1:
    model_name += "-CoT"
model_type = "llama"
for key in llm_template_dict:
    if key in model_name.lower():
        model_type = key

system_prompts_df = pd.read_csv(system_prompts_dir)
system_prompts = system_prompts_df["Prompt"]

if args.hf:
    llm = hfmodel(model_dir, cache_dir, BATCH_SIZE=16)
else:
    if model_type == "command-r":
        llm = LLM(model=model_dir, download_dir=cache_dir, tensor_parallel_size=args.multi_thread)
    else:
        llm = LLM(model=model_dir, download_dir=cache_dir, trust_remote_code=True, tensor_parallel_size=args.multi_thread)  # Create an LLM.

# Here's Mingqian's prompt
#template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{{You will be presented with a role-playing context followed by a multiple-choice question. {role_context} Select only the option number that corresponds to the correct answer for the following question.}}\n\n### Input:\n{{{{{question}}} Provide the number of the correct option without explaining your reasoning.}} \n\n### Response:'''
#flan_template = '''{role_context} {question} Please select the correct answer number:'''
#role_context = "You are a helpful assistant."

metric_dict = {}
benchmark_obj = init_benchmark(name=benchmark)
q_list = benchmark_obj.load_question_list()

user_prompt = benchmark_obj.get_user_prompt(args=args)

for system_prompt in tqdm(system_prompts):
    if system_prompt == "empty":
        system_prompt = ""
    answer_prompts = []
    for q in q_list:
        full_prompt = llm_template_dict[model_type].format(system_prompt=system_prompt, user_prompt=user_prompt.format(question_prompt=q))
        if args.cot == 1:
            full_prompt += " Let's think step by step. "
        answer_prompts.append(full_prompt)
    
    if benchmark == "truthfulqa":
        outputs = llm.generate(answer_prompts, sampling_params=SamplingParams(max_tokens=64))
    elif args.cot != 0:
        outputs = llm.generate(answer_prompts, sampling_params=SamplingParams(max_tokens=512))
    else:
        outputs = llm.generate(answer_prompts)
    
    metric_dict_single = benchmark_obj.eval_question_list(outputs, args=args, save_intermediate=(args.saving_strategy, model_name, system_prompt))
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