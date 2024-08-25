import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import pandas as pd
from tqdm import tqdm
from lib.dataloader import init_benchmark
from lib.modelloader import inference_model
import numpy as np
from sortedcontainers import SortedList
import wandb

wandb.init(project="grips_random")
benchmark_obj_list = [("arc", 1),
                  ("mmlu", 1),
                  ("hellaswag", 1),
                  ("truthfulqa", 1),
                  ("socket_bragging#brag_achievement", 1),
                  ("socket_hahackathon#is_humor", 1),
                  ("socket_tweet_irony", 1),
                  ("socket_sexyn", 1),
                  ("socket_tweet_offensive", 1),
                  ("socket_complaints", 1),
                  ("socket_empathy#empathy_bin", 1),
                  ("socket_stanfordpoliteness", 1),
                  ("socket_rumor#rumor_bool", 1),
                  ("hitom", 1),
                  ("edos_taska", 1),
                  ("ifeval", 1),
                  ("bbh_boolean_expressions", 1),
                  ("bbh_causal_judgement", 1),
                  ("bbh_date_understanding", 1),
                  ("bbh_disambiguation_qa", 1),
                  ("bbh_dyck_languages", 1),
                  ("bbh_formal_fallacies", 1),
                  ("bbh_geometric_shapes", 1),
                  ("bbh_hyperbaton", 1),
                  ("bbh_logical_deduction_five_objects", 1),
                  ("bbh_logical_deduction_seven_objects", 1),
                  ("bbh_logical_deduction_three_objects", 1),
                  ("bbh_movie_recommendation", 1),
                  ("bbh_multistep_arithmetic_two", 1),
                  ("bbh_navigate", 1),
                  ("bbh_object_counting", 1),
                  ("bbh_penguins_in_a_table", 1),
                  ("bbh_reasoning_about_colored_objects", 1),
                  ("bbh_ruin_names", 1),
                  ("bbh_snarks", 1),
                  ("bbh_sports_understanding", 1),
                  ("bbh_temporal_sequences", 1),
                  ("bbh_tracking_shuffled_objects_five_objects", 1),
                  ("bbh_tracking_shuffled_objects_seven_objects", 1),
                  ("bbh_tracking_shuffled_objects_three_objects", 1),
                  ("bbh_web_of_lies", 1),
                  ("bbh_word_sorting", 1),
                  ]
for idx in range(len(benchmark_obj_list)):
    if isinstance(benchmark_obj_list[idx][0], str):
        benchmark_obj_list[idx] = (init_benchmark(name=benchmark_obj_list[idx][0], cot=0), 10)#benchmark_obj_list[idx][1])

benchmark_obj_list_eval = [(benchmark_obj_list[idx][0], None) for idx in range(len(benchmark_obj_list))]

prompt_corpus = pd.read_csv("./data/system_prompts/prompt_corpus_small.csv")
model_obj = inference_model("meta-llama/Meta-Llama-3-8B-Instruct", use_vllm=True, cache_dir="/scratch/qdj_project_owned_root/qdj_project_owned3/shared_data/models/")
eval_metric_name = "avg_score"
full_eval_metric_name = f"{model_obj.model_name}/{eval_metric_name}"

all_prompt_database = {}
if full_eval_metric_name not in all_prompt_database:
    all_prompt_database[full_eval_metric_name] = {}

# initialize heap
all_prompt_heap = SortedList([(-value, key) for key, value in all_prompt_database[full_eval_metric_name].items()])

edit_options = ['del', 'swap', 'sub', 'add']
num_iter = 5000
search_span = 10
sentence_splitter = " /// "

base_prompt = "You are a helpful AI assistant."

if 'sub' in edit_options:
    import torch
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    para_model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
    para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name).to(torch_device).eval()

def run_model_eval(system_prompts, model_obj, benchmark_obj_list, split="all"):
    # Make sure the input format is correct
    system_prompts = np.unique(system_prompts).tolist()
    if not isinstance(benchmark_obj_list, list):
        benchmark_obj_list = [benchmark_obj_list]
    for idx in range(len(benchmark_obj_list)):
        if not isinstance(benchmark_obj_list[idx], tuple):
            benchmark_obj_list[idx] = (benchmark_obj_list[idx], None)
    
    metric_dict = {}
    core_metric_dict = {k:[] for k in system_prompts}
    benchmark_len_list = []

    for benchmark_obj, num_q in benchmark_obj_list:
        q_list, eval_range = benchmark_obj.load_random_question_list(num_q=num_q, split=split)
        benchmark_len_list.append(len(q_list))
        user_prompt = benchmark_obj.get_user_prompt()

        answer_prompts = []
        for system_prompt in system_prompts:
            #answer_prompts = []
            for q in q_list:
                full_prompt = model_obj.get_prompt_template().format(system_prompt=system_prompt, user_prompt=user_prompt.format(question_prompt=q))
                if benchmark_obj.cot == 1:
                    full_prompt += " Let's think step by step. "
                answer_prompts.append(full_prompt)

        full_outputs = model_obj.generate(answer_prompts, max_token_len=512)
            #print(answer_prompts)
            #print(outputs)
            #print("\n\n\n")
        
        for idx, system_prompt in enumerate(system_prompts):
            outputs = full_outputs[(idx)*len(q_list):(idx+1)*len(q_list)]
            metric_dict_single = benchmark_obj.eval_question_list(outputs, save_intermediate=("eval", model_obj.model_name, system_prompt), eval_range=eval_range)
            
            core_metric_dict[system_prompt].append(list(metric_dict_single.values())[0])
            for key, value in metric_dict_single.items():
                if f"{model_obj.model_name}/{key}" not in metric_dict:
                    metric_dict[f"{model_obj.model_name}/{key}"] = {system_prompt: value}
                else:
                    metric_dict[f"{model_obj.model_name}/{key}"][system_prompt] = value
            #metric_dict[system_prompt] = {f"{model_obj.model_name}/{key}": value for key, value in metric_dict_single.items()}

    #core_metric_list = [sum(np.array(core_metric_dict[system_prompt]) * np.array(benchmark_len_list)) / sum(benchmark_len_list) for system_prompt in system_prompts]

    metric_dict[f"{model_obj.model_name}/{eval_metric_name}"] = {}
    for system_prompt in system_prompts:
        metric_dict[f"{model_obj.model_name}/{eval_metric_name}"][system_prompt] = sum(np.array(core_metric_dict[system_prompt]) * np.array(benchmark_len_list)) / np.sum(benchmark_len_list)

    return metric_dict

def rephrase(input_text,num_return_sequences,num_beams):
    batch = para_tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = para_model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = para_tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


# Evaluation
eval_candidates = [""]
metrics_tmp_eval = run_model_eval([candidate.replace(sentence_splitter, "") for candidate in eval_candidates], model_obj, benchmark_obj_list_eval, split="test")
for candidate in eval_candidates:
    for metric_key_tmp in metrics_tmp_eval:
        if "eval_"+metric_key_tmp not in all_prompt_database:
            all_prompt_database["eval_"+metric_key_tmp] = {}
        all_prompt_database["eval_"+metric_key_tmp][candidate] = metrics_tmp_eval[metric_key_tmp][candidate.replace(sentence_splitter, "")]
wandb.log({"test_score": np.mean([all_prompt_database["eval_"+full_eval_metric_name][candidate] for candidate in eval_candidates])}, step=0, commit=True)

for iter_idx in tqdm(range(1, num_iter)):
    
    if len(all_prompt_heap) > 0:
        #weights = np.arange(len(all_prompt_heap), 0, -1)
        exp_scale = 0.5
        weights = np.exp(-np.arange(len(all_prompt_heap)) / exp_scale)
        probabilities = weights / np.sum(weights)
        curr_prompt_idx = np.random.choice(len(all_prompt_heap), p=probabilities)
        curr_prompt = all_prompt_heap.pop(curr_prompt_idx)[1]
    else:
        curr_prompt = base_prompt
    
    edits = np.random.choice(edit_options, search_span)
    candidates = [curr_prompt]
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
                if len(rephrase_candidate.replace(" ", "")) != 0:
                    prompt_component_lst[pos] = rephrase_candidate
                    candidates.append(sentence_splitter.join(prompt_component_lst))
    
    # Deduplicate candidates
    candidates = list(set(candidates))
    
    metrics_tmp = run_model_eval([candidate.replace(sentence_splitter, "") for candidate in candidates], model_obj, benchmark_obj_list, split="train")
    for candidate in candidates:  
        #metrics_tmp = run_model_eval([candidate.replace(sentence_splitter, "")], model_obj, benchmark_obj_list)
        for metric_key_tmp in metrics_tmp:
            if metric_key_tmp not in all_prompt_database:
                all_prompt_database[metric_key_tmp] = {}
            if candidate in all_prompt_database[metric_key_tmp]:
                all_prompt_database[metric_key_tmp][candidate].append(metrics_tmp[metric_key_tmp][candidate.replace(sentence_splitter, "")])
            else:
                all_prompt_database[metric_key_tmp][candidate] = [metrics_tmp[metric_key_tmp][candidate.replace(sentence_splitter, "")]]

        # Record iteration
        if "num_iter" not in all_prompt_database:
            all_prompt_database["num_iter"] = {}
        if candidate in all_prompt_database["num_iter"]:
            all_prompt_database["num_iter"][candidate].append(iter_idx)
        else:
            all_prompt_database["num_iter"][candidate] = [iter_idx]
        
        all_prompt_heap.add((-np.mean(all_prompt_database[full_eval_metric_name][candidate]), candidate))
        assert candidate in all_prompt_database[full_eval_metric_name]
    

    if iter_idx % 5 == 0:
        df_output = pd.DataFrame(all_prompt_database)
        df_output[full_eval_metric_name+"_raw"] = df_output[full_eval_metric_name]
        df_output[full_eval_metric_name] = df_output[full_eval_metric_name].apply(lambda x: np.mean(x))
        wandb.log({"best_score": max(df_output[full_eval_metric_name])}, step=iter_idx, commit=True)
        df_output = df_output.sort_values(by=full_eval_metric_name, ascending=False)
        print(df_output.head(5), flush=True)
        df_output.to_csv("all_prompt_database_grips_random_3.csv")

    # Evaluation
    if iter_idx % 20 == 0:
        eval_candidates = [_item[1] for _item in all_prompt_heap[:10]]
        metrics_tmp_eval = run_model_eval([candidate.replace(sentence_splitter, "") for candidate in eval_candidates], model_obj, benchmark_obj_list_eval, split="test")
        for candidate in eval_candidates:
            for metric_key_tmp in metrics_tmp_eval:
                if "eval_"+metric_key_tmp not in all_prompt_database:
                    all_prompt_database["eval_"+metric_key_tmp] = {}
                all_prompt_database["eval_"+metric_key_tmp][candidate] = metrics_tmp_eval[metric_key_tmp][candidate.replace(sentence_splitter, "")]
        wandb.log({"test_score": np.mean([all_prompt_database["eval_"+full_eval_metric_name][candidate] for candidate in eval_candidates])}, step=iter_idx, commit=True)
wandb.finish()