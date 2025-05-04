import json
import os
import glob
import numpy as np

from transformers import AutoTokenizer
from eval.custom_counter import count_frequencies_with_custom_equal
from eval.math_equivalence import is_equiv

models = []
# alpha_strategies = ["constant", "ppt", "cycle100", "random0.5", "override_annealing"] 
alpha_strategies = ["constant", "constant1.0", "constant0.5", "constant0.25", "constant2.0", "expert_logit_only", "expert_keyword_logits_only"] 

small_size="1.5"
small_base_model=f"Qwen_Qwen2.5-{small_size}B"
small_expert_model=f"deepseek-ai_DeepSeek-R1-Distill-Qwen-{small_size}B"
# large_size="7"
# large_base_model=f"Qwen_Qwen2.5-Math-{large_size}B"
large_size="32"
large_base_model=f"Qwen_Qwen2.5-{large_size}B-Instruct"
large_expert_model=f"deepseek-ai_DeepSeek-R1-Distill-Qwen-{large_size}B"
models += [small_expert_model, large_base_model, large_expert_model]
models += [f"dexperts-{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
models += [f"dexperts-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
models += [f"TwoBody-S{small_size}B-L{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
# models += [f"bexperts-{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]
large_size="72"
large_base_model=f"Qwen_Qwen2.5-Math-{large_size}B-Instruct"
models += [large_base_model]
models += [f"dexperts-{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]



# large_size="70"
# small_size="8"
# large_expert_model=f"deepseek-ai_DeepSeek-R1-Distill-Llama-{large_size}B"
# small_expert_model=f"deepseek-ai_DeepSeek-R1-Distill-Llama-{small_size}B"
# large_base_model=f"meta-llama_Llama-3.3-{large_size}B-Instruct"
# small_base_model=f"meta-llama_Llama-3.1-{small_size}B"
# models += [small_expert_model, large_base_model, large_expert_model]
# models += [f"dexperts-{large_size}B/{alpha_strategy}" for alpha_strategy in alpha_strategies]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B")

def average_token_count(texts):
    """
    Loads a Hugging Face tokenizer and calculates the average number of tokens
    for a list of input strings.

    Args:
        texts (list of str): The input texts to tokenize.
        model_name (str): The Hugging Face model name for the tokenizer.

    Returns:
        float: Average number of tokens per input string.
    """
    total_tokens = 0

    for text in texts:
        tokens = tokenizer.tokenize(text)
        total_tokens += len(tokens)

    avg_tokens = total_tokens / len(texts) if texts else 0
    return avg_tokens

def get_data_items(data_file):
    with open(data_file, 'r') as reader:
        data_items = [json.loads(l) for l in reader]
    for item in data_items:
        item["answer"] = str(item["answer"])
        item["preds"] = list()
    return data_items


def pass_at_k(data_items, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    n = len(data_items[0]['preds'])
    pass_at_k_items = []
    for item in data_items:
        assert n == len(item['preds'])
        c = sum([int(is_equiv(pred, item['answer'])) for pred in item['preds']])
        pass_at_k_items.append(estimator(n, c, k))
    return sum(pass_at_k_items) / len(pass_at_k_items)


def majority(data_items):
    maj_correct_scores = list()
    for item in data_items:
        pred_answer_frequencies = count_frequencies_with_custom_equal(item['preds'], is_equiv)
        voted_answer = pred_answer_frequencies[0][0]
        maj_correct_score = int(is_equiv(voted_answer, item["answer"]))
        maj_correct_scores.append(maj_correct_score)
    return sum(maj_correct_scores) / len(maj_correct_scores)

def main():
    for dataset in ["aime2024", "aime2025", "MATH100"]:
        data_file = f"data/eval/{dataset}/test.jsonl"

        for model in models:
            data_items = get_data_items(data_file)

            pattern = f"results/{dataset}/{model}/*/predictions.jsonl"
            texts = []
            for file_path in glob.glob(pattern):
                with open(file_path, 'r') as reader:
                    pred_items = [json.loads(l) for l in reader]
                assert len(pred_items) == len(data_items)
                for pred_item, data_item in zip(pred_items, data_items):
                    data_item["preds"].append(pred_item["prediction"])
                    texts.append(pred_item["model_output"])
            if glob.glob(pattern): 
                k = len(data_items[0]["preds"])
                pass_at_1 = pass_at_k(data_items, k=1)
                majority_at_k = majority(data_items)
                avg_tokens = average_token_count(texts)
                print(f"{dataset}\t{model}\tpass@1 = {pass_at_1*100:.1f}\tmaj@{k} = {majority_at_k*100:.1f}\tavg_tokens = {avg_tokens:.0f}")



if __name__ == "__main__":
    main()
