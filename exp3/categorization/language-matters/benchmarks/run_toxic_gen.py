import os
import json
import zipfile
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from argparse import ArgumentParser, Namespace
from transformers import AutoTokenizer
from llms import get_llm

isocode2language = {
    'de': 'German',
    'en': 'English',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ru': 'Russian',
    'te': 'Telugu',
    'es': 'Spanish',
    'sw': 'Swahili',
    'zh': 'Chinese'
}


if not os.path.exists('lmsys_toxic'):
    with zipfile.ZipFile('lmsys_toxic.zip', 'r') as zip_ref:
        zip_ref.extractall('./')

def evaluate_dataset(tokenizer, llm, log_filename, src_lang="en", enforcement_text="", temperature=0.6, offset=0):
    if src_lang == 'zh-CN':
        answer_lang = 'Chinese'
    elif '-' in src_lang:
        answer_lang = isocode2language[src_lang.split('-')[-1]]
    else:
        answer_lang = isocode2language[src_lang]


    n_rows_runned = set()
    if log_filename.exists():
        with open(log_filename, "r") as f_in:
            for line in f_in:
                row = json.loads(line)
                n_rows_runned.add(row['conversation_id'])
        print(f"There are {len(n_rows_runned)} rows already runned in {log_filename}. Skipping...")

    os_path = f'lmsys_toxic/{src_lang}'
    dataset = load_dataset(os_path, data_files={'test': 'test.jsonl'})['test']

    for idx, row in enumerate(tqdm(dataset, dynamic_ncols=True)):
        if idx > 2000:
            break

        key = row['conversation_id']
        if key in n_rows_runned:
            continue
        if src_lang != 'en':
            prompt = row['translated']
        else:
            prompt = row['prompt']
        completion_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        if len(enforcement_text):
            if args.prepend_thinking_token:
                completion_text += args.prepend_thinking_token
            completion_text += enforcement_text
            input_str = completion_text
            reasoning, res_info = llm.text_completion(prompt=completion_text, max_tokens=2000, temperature=temperature, stop=["</think>"])
            completion_text += reasoning + f"{args.append_thinking_token}Response in {answer_lang}:\n"
            answer_str, res_info = llm.text_completion(prompt=completion_text, max_tokens=2000, temperature=temperature)
            res_info['input'] = input_str
            res_info['output'] = reasoning+args.append_thinking_token+answer_str
        else:
            _, res_info = llm(prompt=prompt, max_tokens=11000, temperature=temperature)
        res_info['conversation_id'] = row['conversation_id']
        with open(output_path, "a") as f_out:
            f_out.write(json.dumps(res_info) + "\n")

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--series", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--tokenizer_src", default="")
    parser.add_argument("--output_dir", type=Path, default="./log")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lang", default="en", type=str)
    parser.add_argument("--thinking_prefill", type=str, default="")
    parser.add_argument("--prepend_thinking_token", type=str, default=None)
    parser.add_argument("--append_thinking_token", type=str, default="</think>\n")
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_args()
    folder_name = "toxic_bench"
    if args.lang == 'en':
        output_dir = args.output_dir / folder_name / "en"
    else:
        output_dir = args.output_dir / folder_name / args.lang
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = "__".join([
        args.series,
        args.model.split("/")[-1],
    ])
    if args.thinking_prefill:
        output_name += f"__thinking_prefill-{args.thinking_prefill.replace('/','-')}"
    output_path = output_dir / f"{output_name}.jsonl"
    llm = get_llm(args.series, args.model)
    if args.tokenizer_src == "":
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_src)

    evaluate_dataset(tokenizer, llm, output_path,
        enforcement_text=args.thinking_prefill,
        temperature=0.6,
        src_lang=args.lang,
        offset=args.offset
    )
