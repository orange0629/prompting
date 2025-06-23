import os
import json
import zipfile
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from argparse import ArgumentParser, Namespace
from transformers import AutoTokenizer
from llms import get_llm
from prefill_tokens import country_to_prefill

INSTRUCTION_PROMPTS = {
    "en": "{question}\nAnswer: {option}\nIs this answer true or false for this question?",
    "ja": "{question}\n回答: {option}\nこの回答はこの質問に対して真実か偽りか？もし正しいと思えば true、間違っていると思えば false と答えてください。",
    "zh-CN": "{question}\n答案: {option}\n对于这个问题，这个答案是真的还是假的？回答 true 如果你覺得是真的，false 表示答案是錯誤的",
    "ko": "{question}\n답변: {option}\n이 질문에 대한 이 답변은 참인가 거짓인가? 답변이 맞으면 'true', 틀리면 'false'로 응답하세요.",
    "ru": "{question}\nОтвет: {option}\nЯвляется ли этот ответ правдой или ложью для этого вопроса? Ответьте 'true', если ответ правильный, или 'false', если ответ неправильный.",
    "es": "{question}\nRespuesta: {option}\n¿Esta respuesta es verdadera o falsa para esta pregunta? Responde 'true' si la respuesta es correcta, o 'false' si la respuesta es incorrecta.",
    "multi": "{question}\nAnswer: {option}\nIs this answer true or false for this question?"
}

if not os.path.exists('culturalbench'):
    with zipfile.ZipFile('culturalbench.zip', 'r') as zip_ref:
        zip_ref.extractall('./')

def evaluate_dataset(tokenizer, llm, log_filename, prompt_template, lang, args, thinking_prefill="", offset=0):
    n_rows_runned = set()
    if log_filename.exists():
        with open(log_filename, "r") as f_in:
            for line in f_in:
                row = json.loads(line)
                n_rows_runned.add('{}-{}'.format(row['question_idx'], row['data_idx']))
        print(f"There are {len(n_rows_runned)} rows already runned in {log_filename}. Skipping...")
    if lang == 'en':
        dataset = load_dataset("kellycyy/CulturalBench", "CulturalBench-Hard", split="test")
    else:
        os_path = f'culturalbench/{args.lang}'
        dataset = load_dataset(os_path, data_files={'test': 'test.jsonl'})['test']

    for idx, row in enumerate(tqdm(dataset, dynamic_ncols=True)):
        key = '{}-{}'.format(row['question_idx'], row['data_idx'])
        if key in n_rows_runned:
            continue
        question_idx = row['question_idx']
        if question_idx < offset:
            continue
        prompt_question = row['prompt_question']
        prompt_option = row['prompt_option']
        if prompt_option is None: # bruh
            continue
        answer = row['answer'] # True or False
        generation_kwargs = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,            
        }
        prompt = prompt_template.format(question=prompt_question, option=prompt_option)
        if thinking_prefill:
            completion_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            if args.prepend_thinking_token:
                completion_text += args.prepend_thinking_token
            if thinking_prefill == 'multi':
                completion_text += country_to_prefill[row['country']]
            else:
                completion_text += thinking_prefill
            _, res_info = llm.text_completion(prompt=completion_text, **generation_kwargs)
        else:
            _, res_info = llm(prompt=prompt, **generation_kwargs)
            
        res_info['prompt_question'] = row['prompt_question']
        res_info['question_idx'] = row['question_idx']
        res_info['data_idx'] = row['data_idx']
        res_info['prompt_question'] = row['prompt_question']
        res_info['prompt_option'] = row['prompt_option']
        res_info['country'] = row['country']
        res_info["question"] = prompt_question
        res_info["answer"] = str(row["answer"])
        with open(output_path, "a") as f_out:
            f_out.write(json.dumps(res_info) + "\n")

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--series", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--prompt_path", default="")
    parser.add_argument("--tokenizer_src", default="")
    parser.add_argument("--output_dir", type=Path, default="./log")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--prepend_thinking_token", type=str, default=None)
    parser.add_argument("--thinking_prefill", type=str, default="")
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_args()
    if args.prompt_path:
        prompt_path = Path(args.prompt_path)
        prompt_template = prompt_path.read_text()
        folder_name = f"culture_bench__{prompt_path.stem}"
    else:
        prompt_template = "{question}\nAnswer: {option}\nIs this answer true or false for this question?"
        if args.lang in INSTRUCTION_PROMPTS:
            prompt_template = INSTRUCTION_PROMPTS[args.lang]
        folder_name = "culture_bench"

    # if len(args.enforce_text):
    #     enforcement_text = args.enforce_text
    #     enforcement_name = enforcement_text.lower().replace(',','').replace(' ', '-').replace("'", '').replace('/','')
    #     folder_name += '-'+enforcement_name
    output_dir = args.output_dir / folder_name / args.lang
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = "__".join([
        args.series,
        args.model.split("/")[-1],
    ])
    if args.thinking_prefill:
        output_name += f"__thinking_prefill-{args.thinking_prefill}"

    output_path = output_dir / f"{output_name}.jsonl"


    llm = get_llm(args.series, args.model)
    if args.tokenizer_src == "":
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_src)

    evaluate_dataset(tokenizer, llm, output_path,
        prompt_template= prompt_template,
        lang=args.lang,
        thinking_prefill=args.thinking_prefill,
        args=args,
        offset=args.offset
    )
