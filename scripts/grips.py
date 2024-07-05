import pandas as pd
from tqdm import tqdm
from lib.dataloader import init_benchmark
from lib.modelloader import inference_model
import numpy as np
import heapq

prompt_corpus = pd.read_csv("./data/system_prompts/prompt_corpus.csv")
model_obj = inference_model("meta-llama/Meta-Llama-3-8B-Instruct", use_vllm=True, cache_dir="/shared/4/models/")
benchmark_obj = init_benchmark(name="arc", cot=0)
eval_metric_name = "ARC_acc"
full_eval_metric_name = f"{model_obj.model_name}/{eval_metric_name}"

all_prompt_database = {}
if full_eval_metric_name not in all_prompt_database:
    all_prompt_database[full_eval_metric_name] = {}

# initialize heap
all_prompt_heap = [(-value, key) for key, value in all_prompt_database[full_eval_metric_name].items()]
heapq.heapify(all_prompt_heap)

edit_options = ['del', 'swap', 'sub', 'add']
num_iter = 3
search_span = 5
sentence_splitter = " /// "

base_prompt = "You are a helpful AI assistant."

if 'sub' in edit_options:
    import torch
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    para_model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
    para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name).to(torch_device).eval()

def run_model_eval(system_prompts, model_obj, benchmark_obj):
    q_list = benchmark_obj.load_question_list()
    user_prompt = benchmark_obj.get_user_prompt()
    metric_dict = {}

    for system_prompt in system_prompts:
        answer_prompts = []
        for q in q_list:
            full_prompt = model_obj.get_prompt_template().format(system_prompt=system_prompt, user_prompt=user_prompt.format(question_prompt=q))
            if benchmark_obj.cot == 1:
                full_prompt += " Let's think step by step. "
            answer_prompts.append(full_prompt)

        outputs = model_obj.generate(answer_prompts, max_token_len=benchmark_obj.get_max_token_len())
        
        metric_dict_single = benchmark_obj.eval_question_list(outputs, save_intermediate=("eval", model_obj.model_name, system_prompt))
        
        for key, value in metric_dict_single.items():
            if f"{model_obj.model_name}/{key}" not in metric_dict:
                metric_dict[f"{model_obj.model_name}/{key}"] = {system_prompt: value}
            else:
                metric_dict[f"{model_obj.model_name}/{key}"][system_prompt] = value
        #metric_dict[system_prompt] = {f"{model_obj.model_name}/{key}": value for key, value in metric_dict_single.items()}
        
    return metric_dict

def rephrase(input_text,num_return_sequences,num_beams):
    batch = para_tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = para_model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = para_tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

for iter_idx in tqdm(range(num_iter)):
    
    if len(all_prompt_heap) > 0:
        weights = np.arange(len(all_prompt_heap), 0, -1)
        probabilities = weights / np.sum(weights)
        curr_prompt = np.random.choice([tmp[1] for tmp in all_prompt_heap], p=probabilities)
    else:
        curr_prompt = base_prompt
    
    edits = np.random.choice(edit_options, search_span)
    candidates = []
    for edit in edits:
        prompt_component_lst = curr_prompt.split(sentence_splitter)
        if edit == "add" or len(prompt_component_lst) == 0:
            pos = np.random.randint(0, len(prompt_component_lst)+1)
            new_component = np.random.choice(prompt_corpus[prompt_corpus["Catagory"] == np.random.choice(prompt_corpus["Catagory"].unique())]["Prompt"])
            prompt_component_lst.insert(pos, new_component)
            candidates.append(sentence_splitter.join(prompt_component_lst))
        elif edit == "del":
            pos = np.random.randint(0, len(prompt_component_lst))
            prompt_component_lst.pop(pos)
            candidates.append(sentence_splitter.join(prompt_component_lst))
        elif edit == "swap":
            pos1 = np.random.randint(0, len(prompt_component_lst))
            pos2 = np.random.randint(0, len(prompt_component_lst))
            prompt_component_lst[pos1], prompt_component_lst[pos2] = prompt_component_lst[pos2], prompt_component_lst[pos1]
            candidates.append(sentence_splitter.join(prompt_component_lst))
        elif edit == "sub":
            pos = np.random.randint(0, len(prompt_component_lst))
            rephrase_candidates = np.random.choice(rephrase(prompt_component_lst[pos], 10, 10), 3)
            for rephrase_candidate in rephrase_candidates:
                prompt_component_lst[pos] = rephrase_candidate
                candidates.append(sentence_splitter.join(prompt_component_lst))
    
        
    for candidate in candidates:
        if candidate not in all_prompt_database[full_eval_metric_name]:
            metrics_tmp = run_model_eval([candidate.replace(sentence_splitter, "")], model_obj, benchmark_obj)

            for metric_key_tmp in metrics_tmp:
                if metric_key_tmp not in all_prompt_database:
                    all_prompt_database[metric_key_tmp] = {}
                all_prompt_database[metric_key_tmp][candidate] = metrics_tmp[metric_key_tmp][candidate.replace(sentence_splitter, "")]

            assert candidate in all_prompt_database[full_eval_metric_name]
            heapq.heappush(all_prompt_heap, (-all_prompt_database[full_eval_metric_name][candidate], candidate))
    

    if iter_idx % 10 == 0:
        print(all_prompt_heap[:5])
        pd.DataFrame(all_prompt_database).sort_values(by=full_eval_metric_name, ascending=False).to_csv("all_prompt_database.csv", index=False)