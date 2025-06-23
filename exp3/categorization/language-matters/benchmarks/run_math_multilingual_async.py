import json
import zipfile
import asyncio
from tqdm.asyncio import tqdm
from argparse import ArgumentParser, Namespace
from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer

from llms.async_llms import get_async_llm
from run_math_multilingual import LANGUAGES

if not os.path.exists('math_500'):
    with zipfile.ZipFile('math_500.zip', 'r') as zip_ref:
        zip_ref.extractall('./')


def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--series", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--prompt_path", default="")
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_dir", type=Path, default="./log_async")
    parser.add_argument("--prepend_thinking_token", type=str, default=None)
    parser.add_argument("--thinking_prefill", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of tasks to process in each asynchronous batched requests.")
    return parser.parse_args()

async def main():
    args = setup_args()
    assert args.lang in LANGUAGES, f"Language {args.lang} not in {LANGUAGES}"
    if args.lang == 'en':
        ds = load_dataset("HuggingFaceH4/MATH-500")
    else:
        os_path = f'math_500/{args.lang}'
        ds = load_dataset(os_path, data_files={'test': 'test.jsonl'})

    llm = get_async_llm(args.series, args.model, args.base_url)
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
    output_path.write_text("")

    # Inference
    results = []
    total_problems = len(ds["test"])
    if args.debug:
        total_problems = min(10, total_problems)
    
    with tqdm(total=total_problems, desc="Processing math problems") as pbar:
        for i in range(0, total_problems, args.batch_size):
            batch_end = min(i + args.batch_size, total_problems)
            batch_tasks = []
            
            for j in range(i, batch_end):
                row = ds["test"][j]
                q = row["problem"]
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
                    task = llm.text_completion(
                        prompt=completion_text,
                        **generation_kwargs
                    )
                else:
                    task = llm(prompt=prompt, **generation_kwargs)
                batch_tasks.append(task)
            
            try:
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)
                pbar.update(len(batch_results))
            except Exception as e:
                print(f"Error processing batch {i//args.batch_size + 1}: {str(e)}")
                # Continue with next batch even if current batch fails
                continue

    for i, (_, res_info) in enumerate(results):
        res_info["problem"] = ds["test"][i]["problem"]
        res_info["answer"] = ds["test"][i]["answer"]
        with open(output_path, "a") as f_out:
            f_out.write(json.dumps(res_info) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
