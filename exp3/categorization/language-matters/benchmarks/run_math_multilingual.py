import json
from tqdm import tqdm
from colorama import Fore, Style
from argparse import ArgumentParser, Namespace
from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer

from llms import get_llm

LANGUAGES = {
    "en", "zh-CN", "ru", "es", "ja", "ko", "te", "sw",
    "zh-CN-en", "ru-en", "es-en", "ja-en", "ko-en", "te-en", "sw-en"  # back-translated
}

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--series", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--prompt_path", default="")
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_dir", type=Path, default="./log")
    parser.add_argument("--prepend_thinking_token", type=str, default=None)
    parser.add_argument("--thinking_prefill", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def main():
    args = setup_args()
    assert args.lang in LANGUAGES, f"Language {args.lang} not in {LANGUAGES}"
    ds = load_dataset("appier-ai-research/MATH-500-translated", args.lang)
    llm = get_llm(args.series, args.model)
    if args.thinking_prefill:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.prompt_path:
        prompt_path = Path(args.prompt_path)
        prompt_template = prompt_path.read_text()
        folder_name = "MATH-500-"+args.prompt_path.replace('.txt', '').replace('/', '-')
    else:
        prompt_template = "{question}"
        folder_name = "MATH-500"
    output_dir = args.output_dir / folder_name / args.lang
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = "__".join([
        args.model.split("/")[-1],
    ])
    if args.thinking_prefill:
        output_name += f"__thinking_prefill-{args.thinking_prefill}"
    output_path = output_dir / f"{output_name}.jsonl"

    # Inference
    n_rows_runned = 0
    added_problems = set()
    if output_path.exists():
        with open(output_path, "r") as f_in:
            for line in f_in:
                added_problems.add(json.loads(line)['problem'])
        n_rows_runned = len(added_problems)
        print(f"There are {n_rows_runned} rows already runned in {output_path}. Skipping...")

    pbar = tqdm(ds["test"], dynamic_ncols=True)
    for i, row in enumerate(pbar):
        if args.debug and (i >= 10):
            break

        q = row["problem"]
        if q in added_problems:
            continue

        prompt = prompt_template.format(question=q)
        generation_kwargs = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.thinking_prefill:
            completion_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            if args.prepend_thinking_token:
                completion_text += args.prepend_thinking_token
            completion_text += args.thinking_prefill
            _, res_info = llm.text_completion(
                prompt=completion_text,
                **generation_kwargs
            )
        else:
            _, res_info = llm(prompt=prompt, **generation_kwargs)

        res_info["problem"] = row["problem"]
        res_info["answer"] = row["answer"]
        with open(output_path, "a") as f_out:
            f_out.write(json.dumps(res_info) + "\n")

if __name__ == "__main__":
    main()
