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

GLOBAL_SEED = 42  
q_lang = "zh"
MODEL_FULL_NAME = "/shared/4/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
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


MODEL_NAME = "Llama-3.1-8B-Instruct"
RUN_NAME = "20250406"
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
    GPU_IDX_LIST = [0]
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

    benchmark_obj_list = [
                    ("mmlu", 50),
                    ('gsm8k', 50)
                    # ("arc", 1),
                    # ("bbh_date_understanding", 1),
                    # ("bbh_disambiguation_qa", 1),
                    # ("bbh_geometric_shapes", 1),
                    # ("bbh_hyperbaton", 1),
                    # ("bbh_logical_deduction_five_objects", 1),
                    # ("bbh_logical_deduction_seven_objects", 1),
                    # ("bbh_logical_deduction_three_objects", 1),
                    # ("bbh_movie_recommendation", 1),
                    # ("bbh_multistep_arithmetic_two", 1),
                    # ("bbh_object_counting", 1),
                    # ("bbh_reasoning_about_colored_objects", 1),
                    # ("bbh_ruin_names", 1),
                    # ("bbh_snarks", 1),
                    # ("bbh_temporal_sequences", 1),
                    # ("bbh_tracking_shuffled_objects_five_objects", 1),
                    # ("bbh_tracking_shuffled_objects_seven_objects", 1),
                    # ("bbh_tracking_shuffled_objects_three_objects", 1),
                    ]
    for idx in range(len(benchmark_obj_list)):
        name, num_q = benchmark_obj_list[idx]
        benchmark_obj_list[idx] = (init_benchmark(name=name, cot=0), num_q)


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
            # Sample a single question list for all system prompts using a shared seed
            shared_seed = random_seed_lst[0]
            q_list_shared, eval_range_shared = benchmark_obj.load_random_question_list(
                num_q=num_q, split=split, random_seed=shared_seed
            )

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

                translated_q_list_shared = []
                for q in tqdm(q_list_shared, desc=f"Translating to {q_lang}"):
                    if q not in TRANSLATION_CACHE[q_lang] or not TRANSLATION_CACHE[q_lang][q]:
                        translated = translate.translate_text(q, target=q_lang)
                        with lock:
                            TRANSLATION_CACHE[q_lang][q] = translated
                            with open(translate_cache_dir, "w", encoding="utf-8") as f:
                                json.dump(TRANSLATION_CACHE, f, ensure_ascii=False, indent=4)
                    translated_q_list_shared.append(TRANSLATION_CACHE[q_lang][q])
                q_list_shared = translated_q_list_shared

            for idx, system_prompt in enumerate(system_prompts):
                q_list = q_list_shared.copy()
                eval_range = eval_range_shared
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
            
            output_log_path = f"/shared/3/projects/multilingual-system-prompting/full_outputs_log_{MODEL_NAME}_{q_lang}.jsonl"
            os.makedirs(os.path.dirname(output_log_path), exist_ok=True)

            for idx, system_prompt in enumerate(system_prompts):
                outputs = full_outputs[(idx)*len(eval_range_lst[idx]):(idx+1)*len(eval_range_lst[idx])]
                q_list = q_list_shared.copy()
                eval_range = eval_range_shared

                if q_lang != "en":
                    with open(translate_cache_dir, "r", encoding="utf-8") as f_cache:
                        TRANSLATION_CACHE = json.load(f_cache)
                    q_list = [TRANSLATION_CACHE[q_lang].get(q, q) for q in q_list]

                pred_label_list, _ = benchmark_obj.result_list_preprocessing(outputs, answer_identifier=translate.answer_identifiers[q_lang])
                pred_label_list = [label.strip() for label in pred_label_list]
                if eval_range_lst[idx] is None:
                    ground_truth_list = benchmark_obj.true_label_list
                else:
                    ground_truth_list = [benchmark_obj.true_label_list[i] for i in eval_range_lst[idx]]
                ground_truth_list = [label.strip("()") for label in ground_truth_list]
                is_correct_list = [pred_label_list[i] == ground_truth_list[i] for i in range(len(pred_label_list))]


                with open(output_log_path, "a", encoding="utf-8") as f_log:
                    for i, (q, model_out) in enumerate(zip(q_list, outputs)):
                        user_prompt = benchmark_obj.get_user_prompt_new(prompt_type=f"old_{q_lang}")
                        full_prompt_str = PROMPT_TEMPLATE.format(system_prompt=system_prompt, user_prompt=user_prompt.format(question_prompt=q))
                        f_log.write(json.dumps({
                            "benchmark": getattr(benchmark_obj, "benchmark_name", type(benchmark_obj).__name__),
                            "system_prompt": system_prompt,
                            "question": q,
                            "full_input_prompt": full_prompt_str,
                            "model_output": model_out,
                            "pred_label": pred_label_list[i],
                            "ground_truth": ground_truth_list[i],
                            "is_correct": is_correct_list[i]
                        }, ensure_ascii=False) + "\n")


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
            data = json.loads(line.strip())  
            candidate_data.append(data)  

    #for i in tqdm(range(0, len(candidate_prompts), batch_size)):
    for i in tqdm(range(0, len(candidate_data), batch_size)):
        batch_data = candidate_data[i:i + batch_size]
        batch_prompts = [data["prompt"] for data in batch_data]

        random_seed_lst = np.random.randint(0, 2**32, size=len(batch_prompts)).tolist()

        metrics_tmp = run_model_eval_multigpu(
            [prompt.replace(sentence_splitter, " ") for prompt in batch_prompts],
            task_queue, result_queue, benchmark_obj_list, split="train",
            random_seed_lst=[GLOBAL_SEED] * len(batch_prompts),
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