import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import pandas as pd
from tqdm import tqdm
from lib.dataloader import init_benchmark
import numpy as np
import wandb
import multiprocessing
import json
import time
from openai import OpenAI
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import openai
import concurrent.futures

eval_metric_name = "avg_score"

llama3_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
mistral_template = '''<s>[INST]{system_prompt}\n\n{user_prompt}[/INST]'''
qwen_template = '''<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'''

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
        benchmark_obj_list[idx] = (init_benchmark(name=benchmark_obj_list[idx][0], cot=0), 10)

def worker(gpu_id, model_name, task_queue, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from lib.modelloader import inference_model
    model_obj = inference_model(model_name, use_vllm=True, cache_dir="/shared/4/models")
    result_queue.put((gpu_id, "success"))
    while True:
        task = task_queue.get()
        if task is None:
            # print(f"{gpu_id} is stopping.")
            break
        task_idx, task_data = task
        # print(f"{gpu_id} is processing data: {len(task_data)}")
        full_outputs = model_obj.generate(task_data, max_token_len=512)
        result_queue.put((task_idx, full_outputs))
        # print(f"{gpu_id} has finished processing data: {len(task_data)}")

client = openai.OpenAI(
    api_key="",
)

# API调用函数（带重试机制）
def call_chatgpt(prompt: str, max_retries=3) -> str:
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt},
                    ]
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error: {e}, retrying...")
            time.sleep(2)
    return ""

# 组件生成函数
def generate_components(prompt: str, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            response = call_chatgpt(prompt)
            return eval(response[response.find('['):response.rfind(']') + 1])
        except Exception as e:
            retries += 1
            print(f"Error expanding prompt (attempt {retries}/{max_retries}): {type(e).__name__}: {str(e)}")
            time.sleep(3)
    print("Failed to expand prompt after maximum retries.")
    return []

class GeneticRLPrompter:
    
    def __init__(self, init_components, reward_checkpoint):
        self.population = [[c] for c in init_components]  # 初始单组件prompts

        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_checkpoint)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_checkpoint, num_labels=1, torch_dtype=torch.bfloat16, use_flash_attention_2=True
        )
        self.reward_model.to("cuda:0")
        # self.reward_model.eval()
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
            self.reward_model.config.pad_token_id = self.reward_tokenizer.eos_token_id
        
        self.true_scores = {}  # 缓存真实评估结果
        self.component_pool = init_components.copy()
        self.optimize_history = []
        self.step_idx = 0
    
    def evaluate_with_reward_model(self, prompt_lst):
        def tokenize_function(examples):
            return self.reward_tokenizer(examples["prompt"], padding='max_length', truncation=True, max_length=512)
        pred_score_lst = []
        test_dataset_df = pd.DataFrame({"prompt": prompt_lst})
        test_dataset_df["prompt"] = test_dataset_df["prompt"].apply(lambda x: x.replace(" /// ", " "))
        tokenized_test_dataset = Dataset.from_pandas(test_dataset_df).map(tokenize_function, batched=True, remove_columns=["prompt"])
        test_dataloader = DataLoader(tokenized_test_dataset, batch_size=8, collate_fn=lambda batch: {key: torch.stack([torch.tensor(item[key]) for item in batch]) for key in batch[0]})
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                outputs = self.reward_model(input_ids=batch["input_ids"].to("cuda:0"), attention_mask=batch["attention_mask"].to("cuda:0"))
                pred_score_lst.extend(outputs["logits"].reshape(-1).cpu().tolist())
        return pred_score_lst
    
    def run_model_eval_multigpu(self, system_prompts, task_queue, result_queue, benchmark_obj_list, model_name, split="all", saving_strategy="eval", random_seed_lst=None):
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
        
        metric_dict = {}
        core_metric_dict = {k:[] for k in system_prompts}

        for benchmark_obj, num_q in tqdm(benchmark_obj_list):
            user_prompt = benchmark_obj.get_user_prompt_new(prompt_type="old")
            eval_range_lst = []

            answer_prompts = []
            for idx, system_prompt in enumerate(system_prompts):
                q_list, eval_range = benchmark_obj.load_random_question_list(num_q=num_q, split=split, random_seed=random_seed_lst[idx])
                eval_range_lst.append(eval_range)
                for q in q_list:
                    full_prompt = PROMPT_TEMPLATE.format(system_prompt=system_prompt, user_prompt=user_prompt.format(question_prompt=q))
                    answer_prompts.append(full_prompt)

            num_splits = len(GPU_IDX_LIST)
            tasks = [(ii, answer_prompts[ii*len(answer_prompts)//num_splits:(ii+1)*len(answer_prompts)//num_splits]) for ii in range(num_splits)]
            for task in tasks:
                task_queue.put(task)
            
            results = []
            for _ in tasks:
                results.append(result_queue.get())
            results.sort(key=lambda x: x[0])
            full_outputs = []
            for x in results:
                full_outputs += x[1]
            
            for idx, system_prompt in enumerate(system_prompts):
                outputs = full_outputs[(idx)*len(eval_range_lst[idx]):(idx+1)*len(eval_range_lst[idx])]
                metric_dict_single = benchmark_obj.eval_question_list(outputs, save_intermediate=(saving_strategy, model_name, system_prompt), eval_range=eval_range_lst[idx])
                
                core_metric_dict[system_prompt].append(list(metric_dict_single.values())[0])
                for key, value in metric_dict_single.items():
                    if f"{model_name}/{key}" not in metric_dict:
                        metric_dict[f"{model_name}/{key}"] = {system_prompt: value}
                    else:
                        metric_dict[f"{model_name}/{key}"][system_prompt] = value

        metric_dict[f"{model_name}/{eval_metric_name}"] = {}
        for system_prompt in system_prompts:
            metric_dict[f"{model_name}/{eval_metric_name}"][system_prompt] = np.mean(np.array(core_metric_dict[system_prompt]))

        return metric_dict
    
    def reproduce_child(self, parent, partner_lst):
        ACTION_PROB = {
            'add_useful': 0.05,
            'add_useless': 0.025,
            'refine_subset': 0.05,
            'rephrase_subset': 0.05,
            'crossover': 0.825
        }
        
        action = random.choices(
            list(ACTION_PROB.keys()),
            weights=list(ACTION_PROB.values()),
            k=1
        )[0]

        if action == "crossover":
            partner = random.choice(partner_lst)
            child = self.crossover(parent, partner)
            return child
        else:
            return self.random_mutate(parent, action)
    
    def evolutionary_step(self):
        self.step_idx += 1

        # 阶段1：奖励模型评估
        scores = self.evaluate_with_reward_model([" ".join(tmp) for tmp in self.population])
        
        # 阶段2：淘汰后50%
        sorted_population = sorted(zip(scores, self.population), key=lambda x: x[0], reverse=True)
        survivors = [ind for _, ind in sorted_population[:len(scores)//2]]
        
        # 阶段3：生成新后代
        top_10 = survivors[:len(survivors)//10]
        top_50 = survivors[:len(survivors)//2]
        
        # 遗传操作参数
        parents_to_reproduce = random.choices(top_10, k=len(survivors))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.reproduce_child, parent, top_50): parent for parent in parents_to_reproduce}
            new_children = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 更新种群（保持总数量稳定）
        self.population = survivors + new_children
        print(f"Now we got {len(self.population)} after adding {len(new_children)} new children.")
        
        # 阶段4：真实评估采样
        sample_size = min(100, len(self.population))
        sampled = survivors[:5]
        if args.retrain:
            sampled += random.sample(self.population, sample_size)
        real_scores_dict = self.run_model_eval_multigpu([" ".join(tmp) for tmp in sampled], task_queue, result_queue, benchmark_obj_list, model_name=args.model_name, split="train")
        real_scores = [real_scores_dict[f"{args.model_name}/{eval_metric_name}"][tmp] for tmp in sampled]
        self.optimize_history.append({"step": self.step_idx, "train_score": np.mean(real_scores[:5]), "candidates": dict(zip(sampled, real_scores)), "population": self.population})
        wandb.log({"train_score": np.mean(real_scores[:5])}, step=self.step_idx, commit=True)

        # 阶段5：更新奖励模型
        if args.retrain:
            X = [self.extract_features(p) for p in sampled]
            y = real_scores
            self.reward_model.fit(X, y)
    
    # 以下是遗传操作的具体实现示例
    def crossover(self, p1, p2):
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
    
    def random_mutate(self, prompt, action):
        """随机变异：增/删组件"""
        new_components = []
        
        if action == 'add_useful':
            # 基于当前选择生成有用组件
            prompt = f"""You are an expert in optimizing system prompts for LLMs to enhance their general performance. \
Given the following list of system prompt components: {json.dumps(selected)}, generate 1-2 additional components \
that can further improve the LLM's capabilities. Return the result strictly as a Python list of strings. \
No additional explanations or formatting, only return the list."""
            new = generate_components(prompt)
            selected += new
            new_components.extend(new)
            
        elif action == 'add_useless':
            # 添加无用组件
            prompt = f"""Given the following list of system prompt components: {json.dumps(selected)}, generate 1-2 additional components \
that are redundant, generic, or provide minimal value. Examples: ["Answer in English.", "Be polite."]. Return the result strictly \
as a Python list of strings. No additional explanations or formatting, only return the list."""
            new = generate_components(prompt)
            selected += new
            new_components.extend(new)
            
        elif action == 'refine_subset' and len(selected)>=2:
            # 精炼子集为单个组件
            subset = random.sample(selected, min(random.randint(2, 5), len(selected)))
            prompt = f"""Given the following list of sentences: {json.dumps(selected)}, combine these into one concise \
sentence. No additional explanations or formatting, only return a sentence."""
            refined = call_chatgpt(prompt)
            if refined:
                selected = [c for c in selected if c not in subset] + [refined]
                new_components.append(refined)
                
        elif action == 'rephrase_subset' and len(selected)>=1:
            # 重写子集组件
            subset = random.sample(selected, min(random.randint(1, 5), len(selected)))
            rephrased = []
            for c in subset:
                prompt = f"""Rephrase this sentence keeping the same meaning: {c}. \
No additional explanations or formatting, only return a sentence."""
                new_c = call_chatgpt(prompt)
                if new_c:
                    rephrased.append(new_c)
            if rephrased:
                selected = [c for c in selected if c not in subset] + rephrased
                # new_components.extend(rephrased)
        
        # 去重并更新全局组件列表
        unique_new = [c for c in new_components if c not in self.component_pool]
        self.component_pool.extend(unique_new)

        selected = list(set(selected))
        random.shuffle(selected)
        
        return selected
    

# 初始化运行示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Test LLaVA model on Mastodon Dataset")
    parser.add_argument('--model_name', type=str, required=True, help="Path to base model")
    parser.add_argument('--reward_path', type=str, help="Path to the reward model")
    parser.add_argument('--retrain', default=False, action='store_true')
    # parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Mode: train or test")
    # parser.add_argument('--base_model', type=str, required=True, help="Path to base model")
    # parser.add_argument('--model_checkpoint', type=str, help="Path to the model checkpoint")
    # parser.add_argument('--train_path', type=str, help="Path to the training dataset (JSONL)")
    # parser.add_argument('--dev_path', type=str, help="Path to the dev dataset (JSONL)")
    # parser.add_argument('--test_path', type=str, help="Path to the test dataset (JSONL)")
    # parser.add_argument('--output_dir', type=str, help="Path to save the trained model")
    # parser.add_argument('--output_predictions_path', type=str, help="Path to save predictions (for test mode)")
    # parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for training")
    # parser.add_argument('--train_batch_size', type=int, default=1, help="Training batch size")
    # parser.add_argument('--eval_batch_size', type=int, default=1, help="Evaluation batch size")
    # parser.add_argument('--eval_steps', type=int, default=10, help="Evaluation steps")
    # parser.add_argument('--max_model_length', type=int, default=1024, help="Evaluation batch size")

    wandb.init(project="sprig_rlga")

    args = parser.parse_args()

    # Multigpu settings
    GPU_IDX_LIST = [0,1,3]
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    workers = []
    for i in GPU_IDX_LIST:
        process = multiprocessing.Process(target=worker, args=(i, args.model_name, task_queue, result_queue))
        process.start()
        workers.append(process)
    # Verify success
    for _ in GPU_IDX_LIST:
        load_signal = result_queue.get()
        assert load_signal[1] == "success"
    
    from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    from datasets import Dataset
    from torch.utils.data import DataLoader
    import torch

    init_components = [...]  # 这里填入30000个初始组件
    optimizer = GeneticRLPrompter(init_components, "/shared/3/projects/lechen/reward_model/synthetic_modernbert")
    
    for generation in range(100):  # 运行10代
        print(f"Generation {generation+1}")
        optimizer.evolutionary_step()
    
    