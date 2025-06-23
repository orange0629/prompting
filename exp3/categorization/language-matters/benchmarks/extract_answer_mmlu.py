import json
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from argparse import ArgumentParser, Namespace
from tqdm.asyncio import tqdm

prompt_extract = """\
The following text is an LLM's response to a multi choice question answering question:

Text (enclosed in triple quotes): '''{text}'''

Extract the answer from the text (only extract final chosen option A, B, C or D only), and provide it in the following JSON format:
{{"answer": "<single_alphabet_option>"}}"""

client = AsyncOpenAI()

async def chat_completion(prompt: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.0
    )
    return response.choices[0].message.content

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--series", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--use_last_line", action="store_true")
    return parser.parse_args()

async def extract_answer(text: str, thinking_end_token='</think>', use_last_line: bool = False) -> dict:
    """Extract ABCD answer from text using rules or LLM"""
    found_answer = False
    
    if 'Answer:' in text:
        final_answer = text.split('Answer:')[-1].strip()
        if final_answer in "ABCD":
            found_answer = True
            return {"answer_extracted": final_answer, "src": "rule"}
    
    if not found_answer:
        try:
            thinking_and_answer = text.split(thinking_end_token)
            if len(thinking_and_answer) < 2:
                if use_last_line:
                    answer_text = '\n'.join(thinking_and_answer[-1].split('\n')[-2:])
                else:
                    print(f"No {thinking_end_token} found in the text")
                    return {"answer_extracted": "error", "src": "llm"}
            else:
                answer_text = thinking_and_answer[1].strip()
            if len(answer_text) > 1000:
                answer_text = '\n'.join(answer_text.split('\n')[-5:])
            if len(answer_text) > 1000:
                answer_text = answer_text[-500:]
            # if len(answer_text) > 500:
            #     print('hit')
            # Extract answer using LLM
            prompt = prompt_extract.format(text=answer_text.strip())
            res_text = await chat_completion(prompt)
            extracted = json.loads(res_text)
            return {"answer_extracted": extracted["answer"], "src": "llm"}
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return {"answer_extracted": "error", "src": "llm"}

async def main() -> None:
    args = setup_args()
    output_dir = args.input_jsonl.parent / "answer_extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.input_jsonl.name

    # Check for existing progress
    n_rows_runned = set()
    if output_path.exists():
        with open(output_path, "r") as f_in:
            for line in f_in:
                row = json.loads(line)
                n_rows_runned.add(f"{row['qid']}-{row['subject']}")
        print(f"There are {len(n_rows_runned)} rows already runned in {output_path}. Skipping...")

    # Load all rows from input file
    with open(args.input_jsonl) as f_in:
        rows = [json.loads(line) for line in f_in]

    # Create tasks for rows that haven't been processed
    tasks = []
    rows_to_process = []
    for row in rows:
        key = f"{row['qid']}-{row['subject']}"
        if key not in n_rows_runned:
            rows_to_process.append(row)
            tasks.append(extract_answer(row["output"], use_last_line=args.use_last_line))

    if len(tasks) == 0:
        return None

    # Run tasks concurrently using asyncio.gather
    results = await tqdm.gather(*tasks, desc="Checking equalities")
    # results = await asyncio.gather(*tasks)

    # Process results and write to output file
    for row, result in zip(rows_to_process, results):
        row.update(result)
        with open(output_path, "a") as f_out:
            f_out.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
