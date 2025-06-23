import json
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from argparse import ArgumentParser, Namespace
from tqdm.asyncio import tqdm

prompt_extract = """\
The following text is an LLM's response to a true / false question:
Text (enclosed in triple quotes): '''{text}'''
Extract the True / False answer from the text (only answer if its True or False), and provide it in the following JSON format:
{{"answer": "<True/False>"}}"""

client = AsyncOpenAI()

async def chat_completion(prompt: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0
    )
    return response.choices[0].message.content

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--series", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    return parser.parse_args()

async def extract_answer(text: str) -> dict:
    """Extract True/False answer from text using LLM"""
    if '</think>' in text:
        text = text.split('</think>', maxsplit=1)[-1]

    if 'false' in text.lower() and 'true' not in text.lower():
        return {"answer_extracted": "False", "src": "rule"}
    elif 'true' in text.lower() and 'false' not in text.lower():
        return {"answer_extracted": "True", "src": "rule"}

    try:
        prompt = prompt_extract.format(text=text)
        response = await chat_completion(prompt)
        extracted = json.loads(response)
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
                n_rows_runned.add(f"{row['question_idx']}-{row['data_idx']}")
        print(f"There are {len(n_rows_runned)} rows already runned in {output_path}. Skipping...")

    # Load all rows from input file
    with open(args.input_jsonl) as f_in:
        rows = [json.loads(line) for line in f_in]

    # Create tasks for rows that haven't been processed
    tasks = []
    rows_to_process = []
    for row in rows:
        key = f"{row['question_idx']}-{row['data_idx']}"
        if key not in n_rows_runned:
            rows_to_process.append(row)
            tasks.append(extract_answer(row["output"]))

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
