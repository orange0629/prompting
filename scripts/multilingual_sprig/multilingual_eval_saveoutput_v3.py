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
import argparse

GLOBAL_SEED = 42
llama3_template = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
mistral_template = '''<s>[INST]{system_prompt}\n\n{user_prompt}[/INST]'''
qwen_template = '''<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'''
gemma_template = '''<bos><start_of_turn>user\n{system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n'''


RUN_NAME = "20250315"
translate_cache_dir = f'''./cache/translation_cache.json'''
#multiprocessing.set_start_method("spawn", force=True)

def worker(gpu_id, task_queue, result_queue, model_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from lib.modelloader import inference_model
    model_obj = inference_model(model_name, use_vllm=True, cache_dir="/shared/4/models")
    result_queue.put((gpu_id, "success"))
    while True:
        task = task_queue.get()
        if task is None:
            print(f"{gpu_id} is stopping.")
            break
        task_idx, task_data = task
        # print(f"{gpu_id} is processing data: {len(task_data)}")
        full_outputs = model_obj.generate(task_data, max_token_len=4096)
        result_queue.put((task_idx, full_outputs))
        # print(f"{gpu_id} has finished processing data: {len(task_data)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--sys_lang", type=str, default="en")
    parser.add_argument("--task_lang", type=str, default="en")
    parser.add_argument("--benchmark", type=str, default="math500")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[0, 1])
    parser.add_argument("--num_prompts", type=int, default=1000)
    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="/shared/3/projects/multilingual-system-prompting")
    args = parser.parse_args()

    if "llama" in args.model_name.lower():
        PROMPT_TEMPLATE = llama3_template
    elif "mistral" in args.model_name.lower():
        PROMPT_TEMPLATE = mistral_template
    elif "qwen"  in args.model_name.lower():
        PROMPT_TEMPLATE = qwen_template
    elif "gemma" in args.model_name.lower():
        PROMPT_TEMPLATE = gemma_template
    else:
        print("Error, unexpected LLM.")
    
    # Multigpu settings
    GPU_IDX_LIST = args.gpu_ids
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    workers = []
    for i in GPU_IDX_LIST:
        process = multiprocessing.Process(target=worker, args=(i, task_queue, result_queue, args.model_name))
        process.start()
        workers.append(process)
    # Verify success
    for _ in GPU_IDX_LIST:
        load_signal = result_queue.get()
        assert load_signal[1] == "success"

    benchmark_obj_list = [(init_benchmark(name=args.benchmark, cot=0), args.num_questions)]


    def run_model_eval_multigpu(system_prompts, task_queue, result_queue, benchmark_obj_list, split="all", output_log_path=None, q_lang="en"):
        # Make sure the input format is correct
        if not isinstance(benchmark_obj_list, list):
            benchmark_obj_list = [benchmark_obj_list]

        for benchmark_obj, num_q in tqdm(benchmark_obj_list):
            user_prompt = benchmark_obj.get_user_prompt_new(prompt_type=f"old_{q_lang}")
            eval_range_lst = []

            input_prompts = []
            # Sample a single question list for all system prompts using a shared seed
            q_list_shared, eval_range_shared = benchmark_obj.load_random_question_list(
                num_q=num_q, split=split, random_seed=GLOBAL_SEED
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
            for x in results:
                full_outputs += x[1]
            
            os.makedirs(os.path.dirname(output_log_path), exist_ok=True)

            for idx, system_prompt in enumerate(system_prompts):
                outputs = full_outputs[(idx)*len(eval_range_lst[idx]):(idx+1)*len(eval_range_lst[idx])]
                q_list = q_list_shared.copy()
                eval_range = eval_range_shared

                eval_result_dict = benchmark_obj.eval_question_list(outputs, eval_range=eval_range_lst[idx], return_error_idx=True, answer_identifier=translate.answer_identifiers[q_lang], save_intermediate=("eval", "", ""))

                pred_label_list = eval_result_dict["pred_label_list"]
                ground_truth_list = eval_result_dict["true_label_list"]
                is_correct_list = [tmp_idx not in eval_result_dict["error_idx"] for tmp_idx in range(len(pred_label_list))]

                with open(output_log_path, "a", encoding="utf-8") as f_log:
                    for i in range(len(q_list)):
                        f_log.write(json.dumps({
                            "benchmark": benchmark_obj.name,
                            "system_prompt": system_prompt,
                            "question": q_list[i],
                            "full_input_prompt": input_prompts[(idx)*len(eval_range_lst[idx]):(idx+1)*len(eval_range_lst[idx])][i],
                            "model_output": outputs[i],
                            "pred_label": pred_label_list[i],
                            "ground_truth": ground_truth_list[i],
                            "is_correct": is_correct_list[i]
                            # TODO: model_name, input/output lang, sys/task unique id
                        }, ensure_ascii=False) + "\n")

        return


    sentence_splitter = " /// "
    input_dir = f"../../data/system_prompts/generated_prompt_{RUN_NAME}_{args.sys_lang}.jsonl"
    output_log_path = os.path.join(args.output_dir, f"full_outputs_log_{args.model_name.split('/')[-1]}_{args.benchmark}_{args.sys_lang}_{args.task_lang}.jsonl")
    batch_size = 100
    candidate_data = []
    with open(input_dir, "r", encoding="utf-8") as infile:
        for line in infile:
            data = json.loads(line.strip())  
            candidate_data.append(data)  
    candidate_data = candidate_data[:args.num_prompts]

    for i in tqdm(range(0, len(candidate_data), batch_size)):
        batch_data = candidate_data[i:i + batch_size]
        batch_prompts = [data["prompt"] for data in batch_data]

        run_model_eval_multigpu(
            [prompt.replace(sentence_splitter, " ") for prompt in batch_prompts],
            task_queue, result_queue, benchmark_obj_list, split="train",
            q_lang=args.task_lang,
            output_log_path=output_log_path
        )

    
    for _ in workers:
        task_queue.put(None)
    for process in workers:
        process.join()

if __name__ == "__main__":
    main()