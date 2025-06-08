#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import sys
sys.path.append(os.path.abspath("./scripts"))
import pandas as pd
from tqdm import tqdm
from lib.dataloader import init_benchmark
import numpy as np
import wandb
import multiprocessing
import json
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import copy
import concurrent.futures
from filelock import FileLock
import translate
import math
from lib.utils import MinMaxNormalizer
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

GLOBAL_SEED = 42

# Define the metrics we care about
METRIC_COLS = [
    "acc_mean",
    # "acc_var", 
    # "output_tokens_var",
    # "consistency"
]
translate_cache_dir = f'''./scripts/multilingual_sprig/cache/translation_cache.json'''
all_lang_list = ["en", "zh", "es", "fr", "hi"]

llama3_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
mistral_template = '''<s>[INST]{system_prompt}\n\n{user_prompt}[/INST]'''
qwen_template = '''<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'''

llama3_template_task_only = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
mistral_template_task_only = '''<s>[INST]{user_prompt}[/INST]'''
qwen_template_task_only = '''<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'''

benchmark_obj_list = [(init_benchmark(name=tmp_bench_name, cot=0), 50) for tmp_bench_name in ["mmlu_pro", "math500", "unimoral"]]

def worker(gpu_id, model_name, task_queue, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from lib.modelloader import inference_model
    model_obj = inference_model(model_name, use_vllm=True, cache_dir="/scratch/qdj_project_owned_root/qdj_project_owned3/leczhang/models/")
    result_queue.put((gpu_id, "success"))
    while True:
        task = task_queue.get()
        if task is None:
            # print(f"{gpu_id} is stopping.")
            break
        task_idx, task_data = task
        # print(f"{gpu_id} is processing data: {len(task_data)}")
        full_outputs, full_outputs_length = model_obj.generate(task_data, max_token_len=4096, return_length=True)
        result_queue.put((task_idx, full_outputs, full_outputs_length))
        # print(f"{gpu_id} has finished processing data: {len(task_data)}")

def is_all_equal(labels):
    unique_labels = set(labels)
    return 1 if len(unique_labels) == 1 else 0

def compute_metrics(df):
    df_len_var_step1 = (
        df
        .groupby(["benchmark", "prompt", "model", "question_id"])
        .agg(output_len_var=("output_len", "var"))
        .reset_index()
    )

    df_len_var = (
        df_len_var_step1
        .groupby(["benchmark", "model", "prompt"])
        .agg(output_tokens_var=("output_len_var", "mean"))
        .reset_index()
    )

    df_acc_var_len = (
        df
        .groupby(["benchmark", "prompt", "question_lang", "model"])
        .mean(numeric_only=True)
        .reset_index()
        .groupby(["benchmark", "model", "prompt"])
        .agg(
            acc_mean=("is_correct", "mean"),
            acc_var=("is_correct", "var"),
        )
        .reset_index()
    )
    df_consistency = (
        df
        .groupby(["benchmark", "model", "prompt", "question_id"])["pred_label"]
        .apply(is_all_equal)
        .reset_index(name="all_equal_consistency")
        .groupby(["benchmark", "model", "prompt"])["all_equal_consistency"]
        .mean()
        .reset_index(name="consistency")
    )

    df_merged = pd.merge(df_acc_var_len, df_consistency, on=["benchmark", "model", "prompt"], how="left")
    df_merged = pd.merge(df_merged, df_len_var, on=["benchmark", "model", "prompt"], how="left")
    df_merged = df_merged.groupby(["model", "prompt"]).mean(numeric_only=True).reset_index()
    return df_merged

# 以下是遗传操作的具体实现示例
def crossover(p1, p2):
    """杂交操作：随机混合组件"""
    combined = p1 + p2
    random.shuffle(combined)

    # 随机选择一个子代长度
    min_len = min(len(p1), len(p2))
    max_len = len(p1) + len(p2)
    
    # 使用高斯分布随机选取长度，保证大部分接近 max(len(p1), len(p2))，但仍有变化
    mean_len = (len(p1) + len(p2)) / 2  # 取均值作为中心
    std_dev = (max_len - min_len) / 4  # 设定标准差（越大波动越大）
    child_length = int(random.gauss(mean_len, std_dev))
    
    # 限制范围
    child_length = max(1, min(child_length, max_len))
    
    return combined[:child_length]

def random_mutate_batched(selected_batch, action, args):
    if "llama" in args.model_name.lower():
        local_template = llama3_template_task_only
    elif "mistral" in args.model_name.lower():
        local_template = mistral_template_task_only
    elif "qwen"  in args.model_name.lower():
        local_template = qwen_template_task_only

    selected_batch = copy.deepcopy(selected_batch)
    
    if action == 'add_useful':
        input_lst = []
        for selected in selected_batch:
            prompt = f"""You are an expert in optimizing system prompts for LLMs to enhance their general performance. \
Given the following list of system prompt components: {json.dumps(selected)}, generate 1-2 additional components \
that can further improve the LLM's capabilities. Return the result strictly as a Python list of strings. \
No additional explanations or formatting, only return the list."""
            input_lst.append(local_template.replace("{user_prompt}", prompt))
        
        task_queue.put((42, input_lst))
        _, output_lst, _ = result_queue.get()

        for output_idx in range(len(output_lst)):
            output = output_lst[output_idx]
            selected = selected_batch[output_idx]
            try:
                new = eval(output[output.find('['):output.rfind(']') + 1])
            except:
                new = []
            for comp in new:
                insert_idx = random.randint(0, len(selected))
                selected.insert(insert_idx, comp)
    elif action == 'add_useless':
        input_lst = []
        for selected in selected_batch:
            prompt = f"""Given the following list of system prompt components: {json.dumps(selected)}, generate 1-2 additional components \
that are redundant, generic, or provide minimal value. Examples: ["Answer in English.", "Be polite."]. Return the result strictly \
as a Python list of strings. No additional explanations or formatting, only return the list."""
            input_lst.append(local_template.replace("{user_prompt}", prompt))
        
        task_queue.put((42, input_lst))
        _, output_lst, _ = result_queue.get()

        for output_idx in range(len(output_lst)):
            output = output_lst[output_idx]
            selected = selected_batch[output_idx]
            try:
                new = eval(output[output.find('['):output.rfind(']') + 1])
            except:
                new = []
            for comp in new:
                insert_idx = random.randint(0, len(selected))
                selected.insert(insert_idx, comp)
    elif action == 'refine_subset':
        input_lst = []
        subset_lst = []
        for selected in selected_batch:
            subset = random.sample(selected, min(random.randint(2, 5), len(selected)))
            subset_lst.append(subset)
            prompt = f"""Given the following list of sentences: {json.dumps(subset)}, combine these into one concise \
sentence. No additional explanations or formatting, only return a sentence."""
            input_lst.append(local_template.replace("{user_prompt}", prompt))
        
        task_queue.put((42, input_lst))
        _, output_lst, _ = result_queue.get()

        for output_idx in range(len(output_lst)):
            output = output_lst[output_idx]
            selected = selected_batch[output_idx]
            refined = output.strip().split("\n")[0].strip()
            if refined and len(selected) >= 2:
                subset = subset_lst[output_idx]
                for item in list(set(subset)):
                    if item in selected:
                        selected.remove(item)
                    else:
                        print("ERROR!!!")
                        print(f"Item {item} in {refined} not found in selected batch {selected}.\n\n\n")
                insert_idx = random.randint(0, len(selected))
                selected.insert(insert_idx, refined)
            
    elif action == 'rephrase_subset':
        input_lst = []
        for selected in selected_batch:
            item = random.choice(selected)
            prompt = f"""Rephrase this sentence keeping the same meaning: {item}. \
No additional explanations or formatting, only return a sentence."""
            input_lst.append(local_template.replace("{user_prompt}", prompt))
        
        task_queue.put((42, input_lst))
        _, output_lst, _ = result_queue.get()

        for output_idx in range(len(output_lst)):
            output = output_lst[output_idx]
            selected = selected_batch[output_idx]
            new_c = output.strip().split("\n")[0].strip()
            if new_c and len(selected) >= 1:
                item = random.choice(selected)
                selected[selected.index(item)] = new_c
    
    return selected_batch
        

class GeneticRLPrompter:
    
    def __init__(self, init_components, reward_checkpoint):
        self.population = [[c] for c in list(init_components["prompt"])]  # Initial single-component prompts
        print(f"Initialized population with {len(self.population)} components")

        self.normalizer = MinMaxNormalizer.load(os.path.join(reward_checkpoint, "normalizer.json"))
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_checkpoint)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_checkpoint, 
            num_labels=len(METRIC_COLS),  # Now outputs multiple metrics
            torch_dtype=torch.bfloat16, 
        )
        print(f"Loaded reward model from {reward_checkpoint}")
        self.reward_model.to("cuda:0")
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
            self.reward_model.config.pad_token_id = self.reward_tokenizer.eos_token_id
        
        self.component_type_mapping = {}
        for index, row in init_components.iterrows():
            self.component_type_mapping[row['prompt']] = [row["category"]]
        self.component_pool = init_components
        self.optimize_history = []
        self.eval_score_history = []
        self.edit_history = []
        self.step_idx = 0
        print("GeneticRLPrompter initialization complete")
    
    def save_history(self, args):
        print(f"Saving history to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"rlga_history_{args.model_name.split('/')[-1]}_20250420.jsonl"), "w", encoding="utf-8") as f:
            for entry in self.optimize_history:
                f.write(json.dumps(entry) + "\n")
        with open(os.path.join(args.output_dir, f"rlga_score_history_{args.model_name.split('/')[-1]}_20250420.jsonl"), "w", encoding="utf-8") as f:
            for entry in self.eval_score_history:
                f.write(json.dumps(entry) + "\n")
        with open(os.path.join(args.output_dir, f"rlga_edit_history_{args.model_name.split('/')[-1]}_20250420.jsonl"), "w", encoding="utf-8") as f:
            for entry in self.edit_history:
                f.write(json.dumps(entry) + "\n")
        print("History saved successfully")
    
    def evaluate_with_reward_model(self, prompt_lst):
        print(f"Evaluating {len(prompt_lst)} prompts with reward model...")
        def tokenize_function(examples):
            return self.reward_tokenizer(examples["prompt"], padding='max_length', truncation=True, max_length=512)
        
        pred_scores = {metric: [] for metric in METRIC_COLS}
        test_dataset_df = pd.DataFrame({"prompt": prompt_lst})
        test_dataset_df["prompt"] = test_dataset_df["prompt"].apply(lambda x: x.replace(" /// ", " "))
        tokenized_test_dataset = Dataset.from_pandas(test_dataset_df).map(tokenize_function, batched=True, remove_columns=["prompt"])
        test_dataloader = DataLoader(tokenized_test_dataset, batch_size=8, collate_fn=lambda batch: {key: torch.stack([torch.tensor(item[key]) for item in batch]) for key in batch[0]})
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Reward model batch evaluation"):
                outputs = self.reward_model(input_ids=batch["input_ids"].to("cuda:0"), attention_mask=batch["attention_mask"].to("cuda:0"))
                logits = outputs["logits"].to(torch.float32).cpu().numpy()  # Shape: (batch_size, num_metrics)
                for i, metric in enumerate(METRIC_COLS):
                    pred_scores[metric].extend(logits[:, i].tolist())
        
        # Combine scores with weights (you can adjust these weights)
        weights = {
            "acc_mean": 1, #0.5,
            # "acc_var": 0, #-0.25,
            # "output_tokens_var": 0, #-0.125,
            # "consistency": 0 #0.125
        }
        
        combined_scores = []
        for i in range(len(prompt_lst)):
            score = sum(weights[metric] * pred_scores[metric][i] for metric in METRIC_COLS)
            combined_scores.append(score)
        
        print(f"Reward model evaluation complete. Score range: {min(combined_scores):.4f} to {max(combined_scores):.4f}")
        return combined_scores, pred_scores
    
    def run_model_eval_multigpu(self, system_prompts, task_queue, result_queue, benchmark_obj_list, model_name, split="all", random_seed_lst=None):
        
        if random_seed_lst is None:
            random_seed_lst = np.random.randint(0, 2**32, size=len(system_prompts)).tolist()
        
        if "llama" in model_name.lower():
            PROMPT_TEMPLATE = llama3_template
        elif "mistral" in model_name.lower():
            PROMPT_TEMPLATE = mistral_template
        elif "qwen"  in model_name.lower():
            PROMPT_TEMPLATE = qwen_template
        else:
            print("Error")

        eval_item_lst = []

        for benchmark_obj, num_q in tqdm(benchmark_obj_list):
            for q_lang in all_lang_list:
                user_prompt = benchmark_obj.get_user_prompt_new(prompt_type=f"old_{q_lang}")
                eval_range_lst = []

                input_prompts = []
                # For each system prompt, we'll use its corresponding random seed
                q_lists = []
                eval_ranges = []
                
                for idx, _ in enumerate(system_prompts):
                    q_list, eval_range = benchmark_obj.load_random_question_list(
                        num_q=num_q, split=split, random_seed=random_seed_lst[idx]
                    )
                    q_lists.append(q_list)
                    eval_ranges.append(eval_range)
                    eval_range_lst.append(eval_range)

                # Handle translation if needed
                if q_lang != "en":
                    lock_path = translate_cache_dir + '.lock'
                    lock = FileLock(lock_path)
                    if os.path.exists(translate_cache_dir):
                        with open(translate_cache_dir, "r", encoding="utf-8") as f:
                            TRANSLATION_CACHE = json.load(f)
                    else:
                        TRANSLATION_CACHE = {}

                    if q_lang not in TRANSLATION_CACHE:
                        TRANSLATION_CACHE[q_lang] = {}

                    # Translate each set of questions
                    for idx, q_list in enumerate(q_lists):
                        translated_q_list = []
                        for q in tqdm(q_list, desc=f"Translating to {q_lang}"):
                            if q not in TRANSLATION_CACHE[q_lang] or not TRANSLATION_CACHE[q_lang][q]:
                                translated = translate.translate_text(q, target=q_lang)
                                with lock:
                                    TRANSLATION_CACHE[q_lang][q] = translated
                                    with open(translate_cache_dir, "w", encoding="utf-8") as f:
                                        json.dump(TRANSLATION_CACHE, f, ensure_ascii=False, indent=4)
                            translated_q_list.append(TRANSLATION_CACHE[q_lang][q])
                        q_lists[idx] = translated_q_list

                # Create input prompts for each system prompt with its corresponding questions
                for idx, system_prompt in enumerate(system_prompts):
                    q_list = q_lists[idx]
                    for q in q_list:
                        full_prompt = PROMPT_TEMPLATE.format(system_prompt=system_prompt, user_prompt=user_prompt.replace("{question_prompt}", q))
                        input_prompts.append(full_prompt)

                num_splits = len(GPU_IDX_LIST)
                tasks = [(ii, input_prompts[ii*len(input_prompts)//num_splits:(ii+1)*len(input_prompts)//num_splits]) for ii in range(num_splits)]
                for task in tasks:
                    task_queue.put(task)
                
                results = []
                for _ in tasks:
                    results.append(result_queue.get())
                results.sort(key=lambda x: x[0])
                full_outputs = []
                full_outputs_length = []
                for x in results:
                    full_outputs += x[1]
                    full_outputs_length += x[2]
                
                output_log_path = os.path.join(args.output_dir, f"sprig_outputs_log_{args.model_name.split('/')[-1]}.jsonl")
                os.makedirs(os.path.dirname(output_log_path), exist_ok=True)

                for idx, system_prompt in enumerate(system_prompts):
                    q_list = q_lists[idx]
                    eval_range = eval_ranges[idx]
                    
                    # Calculate start and end indices for this system prompt's outputs
                    start_idx = sum(len(eval_ranges[i]) for i in range(idx))
                    end_idx = start_idx + len(eval_range)
                    
                    outputs = full_outputs[start_idx:end_idx]
                    outputs_length = full_outputs_length[start_idx:end_idx]

                    eval_result_dict = benchmark_obj.eval_question_list(outputs, eval_range=eval_range, return_error_idx=True, answer_identifier=translate.answer_identifiers[q_lang], save_intermediate=("eval", "", ""))

                    pred_label_list = eval_result_dict["pred_label_list"]
                    ground_truth_list = eval_result_dict["true_label_list"]
                    is_correct_list = [tmp_idx not in eval_result_dict["error_idx"] for tmp_idx in range(len(pred_label_list))]

                    with open(output_log_path, "a", encoding="utf-8") as f_log:
                        for i in range(len(q_list)):
                            eval_item = {
                                "benchmark": benchmark_obj.name,
                                "model": args.model_name.split('/')[-1],
                                "prompt": system_prompt,
                                "question_lang": q_lang,
                                "question": q_list[i],
                                "question_id": eval_range[i],
                                "full_input_prompt": input_prompts[start_idx + i],
                                "model_output": outputs[i],
                                "pred_label": pred_label_list[i],
                                "ground_truth": ground_truth_list[i],
                                "is_correct": is_correct_list[i],
                                "split": split,
                                "output_len": outputs_length[i],
                                # TODO: model_name, input/output lang, sys/task unique id
                            }
                            eval_item_lst.append(eval_item)
                            f_log.write(json.dumps(eval_item, ensure_ascii=False) + "\n")
        
        raw_result_df = pd.DataFrame(eval_item_lst)
        metric_result_df = compute_metrics(raw_result_df)
        metric_result_df = metric_result_df.sort_values('prompt', key=lambda x: [system_prompts.index(i) for i in x]).reset_index(drop=True)
        normalized_metric_result_df = metric_result_df.copy()
        for m in METRIC_COLS:
            normalized_metric_result_df[m] = metric_result_df[m].apply(lambda x: self.normalizer.normalize(m, x))
        # normalize and weighted sum
        metric_result_df["overall_score"] = normalized_metric_result_df["acc_mean"]# + 0.25 * (1 - normalized_metric_result_df["acc_var"]) + 0.125 * (1 - normalized_metric_result_df["output_tokens_var"]) + 0.125 * normalized_metric_result_df["consistency"]) / 4
        print(metric_result_df, flush=True)
        return metric_result_df

    def random_mutate(self, selected, action):
        """随机变异：增/删组件"""
        selected = selected.copy()
        new_components = []
        if action == 'add_random':
            # 添加随机组件
            comp = random.sample(list(self.component_pool["prompt"]), k=1)[0]
            insert_idx = random.randint(0, len(selected))
            selected.insert(insert_idx, comp)
        
        elif action == 'swap' and len(selected) >= 2:
            idx1, idx2 = random.sample(range(len(selected)), 2)
            selected[idx1], selected[idx2] = selected[idx2], selected[idx1]
        
        elif action == 'delete' and selected:
            remove_idx = random.randint(0, len(selected) - 1)
            selected.pop(remove_idx)
        
        return selected

    def evolutionary_step(self, args):
        self.step_idx += 1
        print(f"\n=== Starting evolutionary step {self.step_idx} ===")
        print(f"Current population size: {len(self.population)}")

        # Stage 1: Reward model evaluation with multiple metrics
        combined_scores, metric_scores = self.evaluate_with_reward_model([" ".join(tmp) for tmp in self.population])
        
        # Stage 2: Eliminate bottom 50% based on combined score
        sorted_population = sorted(zip(combined_scores, self.population), key=lambda x: x[0], reverse=True)
        survivors = [ind for _, ind in sorted_population[:len(combined_scores)//2]]
        print(f"Selected {len(survivors)} survivors out of {len(self.population)} individuals")

        # Stage 2.5: Evaluate top 10 on dev set
        top_10_survivors = survivors[:10]
        print(f"Evaluating top {len(top_10_survivors)} survivors on test set")
        dev_scores_df = self.run_model_eval_multigpu(
            [" ".join(tmp) for tmp in top_10_survivors], 
            task_queue, 
            result_queue, 
            benchmark_obj_list, 
            model_name=args.model_name.split("/")[-1], 
            split="test", 
            random_seed_lst=[42 for _ in range(len(top_10_survivors))]
        )
        
        # Log detailed metrics for each survivor
        for metric in METRIC_COLS + ["overall_score"]:
            wandb.log({f"test_{metric}": np.mean(dev_scores_df[metric]), "global_step": self.step_idx})
        dev_scores_df["step"] = self.step_idx
        dev_scores_df["split"] = "test"
        self.eval_score_history.extend(dev_scores_df.to_dict(orient="records"))
        print(f"Step {self.step_idx}: Test evaluation complete. Average score: {np.mean(dev_scores_df['overall_score']):.4f}")

        # Stage 3: Generate new offspring
        top_10 = survivors[:len(survivors)//10]
        top_50 = survivors[:len(survivors)//2]
        target_num_new = len(survivors)
        print(f"Generating {target_num_new} new offspring")
        
        # Genetic operation parameters
        survivors_set = set([" ".join(tmp_lst) for tmp_lst in survivors])
        new_children_set, new_children = set(), []

        ACTION_PROB = {
            'add_useful': 0.025,
            'add_useless': 0.01,
            'add_random': 0.20,
            'refine_subset': 0.025,
            'rephrase_subset': 0.025,
            'swap': 0.05,
            'delete': 0.05,
            'crossover': 0.615
        }

        # Do cumulative expansion
        expensive_actions = ["add_useful", "add_useless", "refine_subset", "rephrase_subset"]
        pbar = tqdm(total=target_num_new, desc="Generating children")
        cummulative_prob = 0

        for action in expensive_actions:
            cummulative_prob += ACTION_PROB[action]
            tmp_num_duplicants = 0
            needed = int(cummulative_prob * target_num_new) - len(new_children)
            parents_to_reproduce = random.choices(top_10, k=needed)

            child_result_lst = random_mutate_batched(parents_to_reproduce, action, args)
            for child_idx in range(len(child_result_lst)):
                child = child_result_lst[child_idx]
                child_str = " ".join(child)
                if child_str not in new_children_set and child_str not in survivors_set:
                    new_children_set.add(child_str)
                    new_children.append(child)
                    self.edit_history.append({"step": self.step_idx, "new_child": child, "parent": [parents_to_reproduce[child_idx]], "action": action})
                    pbar.update(1)
                else:
                    tmp_num_duplicants += 1
            print(f"Duplicated {tmp_num_duplicants} prompts for action {action}.")

        while len(new_children) < target_num_new:
            action = random.choices(
                list(ACTION_PROB.keys()),
                weights=list(ACTION_PROB.values()),
                k=1
            )[0]
            if action in expensive_actions:
                continue
            parent = None
            if action == "crossover":
                parent1 = random.choice(top_10)
                parent2 = random.choice(top_50)
                child = crossover(parent1, parent2)
            else:
                parent = random.choice(top_10)
                child = self.random_mutate(parent, action)
            
            child_str = " ".join(child)
            if child_str not in new_children_set and child_str not in survivors_set:
                new_children_set.add(child_str)
                new_children.append(child)
                if parent:
                    self.edit_history.append({"step": self.step_idx, "new_child": child, "parent": [parent], "action": action})
                else:
                    self.edit_history.append({"step": self.step_idx, "new_child": child, "parent": [parent1, parent2], "action": action})
                pbar.update(1)
        pbar.close()

        # 更新种群（保持总数量稳定）
        self.population = survivors + new_children
        print(f"Now we got {len(self.population)} after adding {len(new_children)} new children.")
        
        # Stage 4: Real evaluation sampling
        sample_size = min(100, len(self.population))
        sampled = survivors[:10]
        if args.retrain:
            sampled += random.sample(self.population, sample_size)
        print(f"Evaluating {len(sampled)} sampled individuals on train set")
        real_scores_df = self.run_model_eval_multigpu([" ".join(tmp) for tmp in sampled], task_queue, result_queue, benchmark_obj_list, model_name=args.model_name.split("/")[-1], split="train_small", random_seed_lst=np.random.randint(0, 2**32, size=len(sampled)).tolist())
        real_scores_df["step"] = self.step_idx
        real_scores_df["split"] = "train"
        self.eval_score_history.extend(real_scores_df.to_dict(orient="records"))
        wandb.log({"train_score": np.mean(real_scores_df["overall_score"]), "global_step": self.step_idx})
        print(f"Step {self.step_idx}: Train Score: {np.mean(real_scores_df['overall_score'])}", flush=True)
        self.optimize_history.append({"step": self.step_idx, "train_score": np.mean(real_scores_df['overall_score']), "candidates": dict(zip([" /// ".join(tmp) for tmp in sampled], list(real_scores_df['overall_score']))), "population": self.population})

        # Stage 5: Update reward model if retraining is enabled
        if args.retrain:
            print(f"Retraining reward model at step {self.step_idx}")
            from reward_modeling.main_multi_metric_one import PairwiseTrainer, RewardConfig, RewardDataCollatorWithPadding, build_pairs, make_tok_pair_fn
            from sklearn.model_selection import train_test_split
            
            prompt_template = "{system_prompt}"
            full_replay = [x for x in self.eval_score_history if x["split"] == "train" and x["step"] < self.step_idx]
            print(f"Using {len(full_replay)} historical samples for retraining")
            
            # Prepare training data with multiple metrics
            data = real_scores_df.to_dict(orient="records") + random.sample(full_replay, min(50, len(full_replay)))
            train_rows, dev_rows = train_test_split(data, test_size=0.4, random_state=GLOBAL_SEED)
            print(f"Training set: {len(train_rows)}, Dev set: {len(dev_rows)}")

            prompt_template = "{system_prompt}"

            train_ds = build_pairs(train_rows, prompt_template, self.normalizer).select(range(len(train_rows)*10))
            dev_ds = build_pairs(dev_rows, prompt_template, self.normalizer).select(range(len(dev_rows)*10))
            self.reward_model.train()
            train_ds = train_ds.map(
                make_tok_pair_fn(self.reward_tokenizer, args.max_length if hasattr(args, 'max_length') else 512),
                batched=True,
                remove_columns=train_ds.column_names,
            )
            dev_ds = dev_ds.map(
                make_tok_pair_fn(self.reward_tokenizer, args.max_length if hasattr(args, 'max_length') else 512),
                batched=True,
                remove_columns=dev_ds.column_names,
            )

            
            # ─────────────────────────── 3. training args ──────────────────────
            train_args = TrainingArguments(
                output_dir=args.cache_dir,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                eval_strategy="steps",
                eval_steps=10,
                num_train_epochs=1,
                save_strategy="steps",
                save_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                save_total_limit=2,
                gradient_accumulation_steps=1,
                gradient_checkpointing=True,
                bf16=True,
                logging_strategy="steps",
                logging_steps=1,
                report_to="wandb", 
                remove_unused_columns=False,
                learning_rate=5e-6,
                lr_scheduler_type="cosine",
            )

            # ─────────────────────────── 4. initialise trainer ────────────────
            trainer = PairwiseTrainer(
                model=self.reward_model,
                args=train_args,
                tokenizer=self.reward_tokenizer,
                train_dataset=train_ds,                            # already tokenised
                eval_dataset=dev_ds,
                data_collator=RewardDataCollatorWithPadding(self.reward_tokenizer),
            )

            trainer.train()
            self.reward_model.eval()
            print("Reward model retraining complete")

        print(f"=== Evolutionary step {self.step_idx} complete ===\n")


# 初始化运行示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Test LLaVA model on Mastodon Dataset")
    parser.add_argument('--model_name', type=str, required=True, help="Path to base model")
    parser.add_argument('--reward_path', type=str, help="Path to the reward model")
    parser.add_argument('--retrain', default=False, action='store_true')
    parser.add_argument('--output_dir', type=str, help="Path to save the trained model")
    parser.add_argument('--cache_dir', type=str, help="Path to save the trained model")
    # parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Mode: train or test")
    # parser.add_argument('--base_model', type=str, required=True, help="Path to base model")
    # parser.add_argument('--model_checkpoint', type=str, help="Path to the model checkpoint")
    # parser.add_argument('--train_path', type=str, help="Path to the training dataset (JSONL)")
    # parser.add_argument('--dev_path', type=str, help="Path to the dev dataset (JSONL)")
    # parser.add_argument('--test_path', type=str, help="Path to the test dataset (JSONL)")
    # parser.add_argument('--output_predictions_path', type=str, help="Path to save predictions (for test mode)")
    # parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for training")
    # parser.add_argument('--train_batch_size', type=int, default=1, help="Training batch size")
    # parser.add_argument('--eval_batch_size', type=int, default=1, help="Evaluation batch size")
    # parser.add_argument('--eval_steps', type=int, default=10, help="Evaluation steps")
    # parser.add_argument('--max_model_length', type=int, default=1024, help="Evaluation batch size")

    print("Starting RLGA experiment")
    wandb.init(project="sprig_multilingual")
    print(f"Wandb initialized with project: sprig_multilingual")

    args = parser.parse_args()
    print(f"Arguments: {args}")

    # Multigpu settings
    GPU_IDX_LIST = [1,2,3,4,5,6,7]
    print(f"Using GPUs: {GPU_IDX_LIST}")
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    workers = []
    for i in GPU_IDX_LIST:
        process = multiprocessing.Process(target=worker, args=(i, args.model_name, task_queue, result_queue))
        process.start()
        workers.append(process)
        print(f"Started worker on GPU {i}")
    # Verify success
    for _ in GPU_IDX_LIST:
        load_signal = result_queue.get()
        assert load_signal[1] == "success"
        print(f"Worker on GPU {load_signal[0]} successfully loaded model")
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # with open("./data/system_prompts/dynamic_components_20250127.json", "r", encoding="utf-8") as file:
    #     init_components = json.load(file)
    print("Loading prompt components data")
    init_components = pd.read_json("./data/system_prompts/generated_prompt_components_20250315.jsonl", lines=True)
    print(f"Loaded {len(init_components)} prompt components")
    
    optimizer = GeneticRLPrompter(init_components, args.reward_path)
    
    for generation in range(25):  # 运行50代
        print(f"\n=================== Generation {generation+1}/25 ===================")
        optimizer.evolutionary_step(args)
        optimizer.save_history(args)
    
    print("RLGA experiment completed successfully!")
    
    