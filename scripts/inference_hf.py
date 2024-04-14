from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from lib.dataloader import init_benchmark
import argparse
import torch

cache_dir= "/scratch/jurgens_owned_root/jurgens_owned1/shared_data/models/"
BATCH_SIZE = 16
llama_template = '''[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{user_prompt}[/INST]Answer:'''
mixtral_template = '''<s> [INST] {system_prompt}\n{user_prompt} [/INST] Answer:'''
llm_template_dict = {"llama": llama_template, "mixtral": mixtral_template}


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
model_type = "llama"
for key in llm_template_dict:
    if key in model_name.lower():
        model_type = key

system_prompts_df = pd.read_csv(system_prompts_dir)
system_prompts = system_prompts_df["Prompt"]

# Create an LLM.
tokenizer = AutoTokenizer.from_pretrained(model_dir, 
                                           cache_dir=cache_dir,
                                           padding_side='left',
                                           )
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_skip_modules=["mamba"])
model = AutoModelForCausalLM.from_pretrained(model_dir,
                                             cache_dir=cache_dir,
                                             trust_remote_code=True,
                                             device_map="auto",
                                             torch_dtype=torch.bfloat16,
                                             attn_implementation="flash_attention_2",
                                             quantization_config=quantization_config)

user_prompt = "The following is a multiple choice question (with answers). Reply with only the option letter.\n{question_prompt}"


metric_dict = {}
benchmark_obj = init_benchmark(name=benchmark)
q_list = benchmark_obj.load_question_list()

for system_prompt in tqdm(system_prompts):
    answer_prompts = []
    for q in q_list:
        full_prompt = llm_template_dict[model_type].format(system_prompt=system_prompt, user_prompt=user_prompt.format(question_prompt=q))
        answer_prompts.append(full_prompt)
    
    # Generate results
    outputs = []
    for idx in tqdm(range(0, len(answer_prompts), BATCH_SIZE)):
        ques_batch = answer_prompts[idx:(idx+BATCH_SIZE)]
        ques_batch_tokenized = tokenizer(ques_batch, return_tensors='pt', truncation=True, max_length=512, padding=True)
        answ_ids = model.generate(**ques_batch_tokenized.to('cuda'), max_new_tokens=30, pad_token_id=tokenizer.pad_token_id)
        outputs.extend(tokenizer.batch_decode(answ_ids, skip_special_tokens=True))

    metric_dict_single = benchmark_obj.eval_question_list(outputs, vllm=False, save_intermediate=(True, f"{model_name}/{system_prompt}"))
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
