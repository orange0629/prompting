import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import sys
sys.path.append(os.path.abspath("../"))

import pandas as pd
from tqdm import tqdm
from lib.dataloader import init_benchmark
import numpy as np
import wandb
import multiprocessing
import json
import time
from openai import OpenAI
from filelock import FileLock
import translate


q_lang = "zh"
MODEL_FULL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llama3_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
mistral_template = '''<s>[INST]{system_prompt}\n\n{user_prompt}[/INST]'''
qwen_template = '''<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'''

if "llama" in MODEL_FULL_NAME.lower():
    PROMPT_TEMPLATE = llama3_template
elif "mistral" in MODEL_FULL_NAME.lower():
    PROMPT_TEMPLATE = mistral_template
elif "qwen"  in MODEL_FULL_NAME.lower():
    PROMPT_TEMPLATE = qwen_template
else:
    print("Error")


MODEL_NAME = MODEL_FULL_NAME.split("/")[-1]
RUN_NAME = "20250315"
translate_cache_dir = f'''./cache/translation_cache.json'''
#multiprocessing.set_start_method("spawn", force=True)

def worker(gpu_id, task_queue, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from lib.modelloader import inference_model
    model_obj = inference_model(MODEL_FULL_NAME, use_vllm=True, cache_dir="/shared/4/models")
    result_queue.put((gpu_id, "success"))
    while True:
        task = task_queue.get()
        if task is None:
            print(f"{gpu_id} is stopping.")
            break
        task_idx, task_data = task
        # print(f"{gpu_id} is processing data: {len(task_data)}")
        full_outputs = model_obj.generate(task_data, max_token_len=512)
        result_queue.put((task_idx, full_outputs))
        # print(f"{gpu_id} has finished processing data: {len(task_data)}")

def main():
    # Multigpu settings
    GPU_IDX_LIST = [0,1,2,4]
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    workers = []
    for i in GPU_IDX_LIST:
        process = multiprocessing.Process(target=worker, args=(i, task_queue, result_queue))
        process.start()
        workers.append(process)
    # Verify success
    for _ in GPU_IDX_LIST:
        load_signal = result_queue.get()
        assert load_signal[1] == "success"

    benchmark_obj_list = [("arc", 1),
                    ("mmlu", 1),
                    ("bbh_date_understanding", 1),
                    ("bbh_disambiguation_qa", 1),
                    ("bbh_geometric_shapes", 1),
                    ("bbh_hyperbaton", 1),
                    ("bbh_logical_deduction_five_objects", 1),
                    ("bbh_logical_deduction_seven_objects", 1),
                    ("bbh_logical_deduction_three_objects", 1),
                    ("bbh_movie_recommendation", 1),
                    ("bbh_multistep_arithmetic_two", 1),
                    ("bbh_object_counting", 1),
                    ("bbh_reasoning_about_colored_objects", 1),
                    ("bbh_ruin_names", 1),
                    ("bbh_snarks", 1),
                    ("bbh_temporal_sequences", 1),
                    ("bbh_tracking_shuffled_objects_five_objects", 1),
                    ("bbh_tracking_shuffled_objects_seven_objects", 1),
                    ("bbh_tracking_shuffled_objects_three_objects", 1),
                    ]
    for idx in range(len(benchmark_obj_list)):
        if isinstance(benchmark_obj_list[idx][0], str):
            benchmark_obj_list[idx] = (init_benchmark(name=benchmark_obj_list[idx][0], cot=0), 20)#benchmark_obj_list[idx][1])

    eval_metric_name = "avg_score"

    def run_model_eval_multigpu(system_prompts, task_queue, result_queue, benchmark_obj_list, split="all", saving_strategy="eval", random_seed_lst=None, q_lang="en"):
        # Make sure the input format is correct
        if random_seed_lst is None:
            random_seed_lst = np.random.randint(0, 2**32, size=len(system_prompts)).tolist()
        if not isinstance(benchmark_obj_list, list):
            benchmark_obj_list = [benchmark_obj_list]
        for idx in range(len(benchmark_obj_list)):
            if not isinstance(benchmark_obj_list[idx], tuple):
                benchmark_obj_list[idx] = (benchmark_obj_list[idx], None)
        
        metric_dict = {}
        core_metric_dict = {k:[] for k in system_prompts}

        for benchmark_obj, num_q in tqdm(benchmark_obj_list):
            user_prompt = benchmark_obj.get_user_prompt_new(prompt_type=f"old_{q_lang}")
            eval_range_lst = []

            answer_prompts = []
            for idx, system_prompt in enumerate(system_prompts):
                q_list, eval_range = benchmark_obj.load_random_question_list(num_q=num_q, split=split, random_seed=random_seed_lst[idx])
                # Specify Question language
                for q_idx in tqdm(range(len(q_list)), desc=f"Translating to {q_lang}: "):
                    lock_path = translate_cache_dir + '.lock'
                    lock = FileLock(lock_path)
                    if q_lang == "en":
                        continue
                    try:
                        assert isinstance(TRANSLATION_CACHE[q_lang][q_list[q_idx]], str)
                    except:
                        with lock:
                            if os.path.exists(translate_cache_dir):
                                with open(translate_cache_dir, "r", encoding="utf-8") as f:
                                    TRANSLATION_CACHE = json.load(f)
                            else:
                                TRANSLATION_CACHE = {}
                        if q_lang != "en":
                            if q_lang not in TRANSLATION_CACHE:
                                TRANSLATION_CACHE[q_lang] = {}
                            if q_list[q_idx] not in TRANSLATION_CACHE[q_lang] or (not TRANSLATION_CACHE[q_lang][q_list[q_idx]]):
                                tmp_translate_result = translate.translate_text(q_list[q_idx], target=q_lang)
                                with lock:
                                    if os.path.exists(translate_cache_dir):
                                        with open(translate_cache_dir, "r", encoding="utf-8") as f:
                                            TRANSLATION_CACHE = json.load(f)
                                    else:
                                        TRANSLATION_CACHE = {}
                                    if q_lang not in TRANSLATION_CACHE:
                                        TRANSLATION_CACHE[q_lang] = {}
                                    TRANSLATION_CACHE[q_lang][q_list[q_idx]] = tmp_translate_result
                                    with open(translate_cache_dir, "w", encoding="utf-8") as f:
                                        json.dump(TRANSLATION_CACHE, f, ensure_ascii=False, indent=4)
                        
                        q_list[q_idx] = TRANSLATION_CACHE[q_lang][q_list[q_idx]]
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
                model_name = MODEL_NAME
                metric_dict_single = benchmark_obj.eval_question_list(outputs, save_intermediate=(saving_strategy, model_name, system_prompt), eval_range=eval_range_lst[idx], answer_identifier=translate.answer_identifiers[q_lang])
                
                core_metric_dict[system_prompt].append(list(metric_dict_single.values())[0])
                for key, value in metric_dict_single.items():
                    if f"{model_name}/{key}" not in metric_dict:
                        metric_dict[f"{model_name}/{key}"] = {system_prompt: value}
                    else:
                        metric_dict[f"{model_name}/{key}"][system_prompt] = value
                #metric_dict[system_prompt] = {f"{model_obj.model_name}/{key}": value for key, value in metric_dict_single.items()}

        #core_metric_list = [sum(np.array(core_metric_dict[system_prompt]) * np.array(benchmark_len_list)) / sum(benchmark_len_list) for system_prompt in system_prompts]

        metric_dict[f"{model_name}/{eval_metric_name}"] = {}
        for system_prompt in system_prompts:
            metric_dict[f"{model_name}/{eval_metric_name}"][system_prompt] = np.mean(np.array(core_metric_dict[system_prompt]))

        return metric_dict


    sentence_splitter = " /// "
    input_dir = f"../../data/system_prompts/generated_prompt_{RUN_NAME}_{q_lang}.jsonl"
    output_dir = input_dir.replace(".jsonl", "_scored.jsonl")
    batch_size = 100
    candidate_data = []
    with open(input_dir, "r", encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line.strip())  # 解析 JSONL 行
            candidate_data.append(data)  # 保持完整结构

    #for i in tqdm(range(0, len(candidate_prompts), batch_size)):
    for i in tqdm(range(0, len(candidate_data), batch_size)):
        batch_data = candidate_data[i:i + batch_size]
        batch_prompts = [data["prompt"] for data in batch_data]

        random_seed_lst = np.random.randint(0, 2**32, size=len(batch_prompts)).tolist()

        metrics_tmp = run_model_eval_multigpu(
            [prompt.replace(sentence_splitter, " ") for prompt in batch_prompts],
            task_queue, result_queue, benchmark_obj_list, split="train",
            random_seed_lst=random_seed_lst,
            q_lang=q_lang,
        )

        with open(output_dir, "a", encoding="utf-8") as f:
            for prompt_idx, data in enumerate(batch_data):
                output_dict = data.copy()
                for metric_key_tmp in metrics_tmp:
                    output_dict[metric_key_tmp] = metrics_tmp[metric_key_tmp][data["prompt"].replace(sentence_splitter, " ")]
                output_dict["random_seed"] = random_seed_lst[prompt_idx]
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
    
    for _ in workers:
        task_queue.put(None)
    for process in workers:
        process.join()

if __name__ == "__main__":
    main()