import json
from pathlib import Path
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    return parser.parse_args()

def main():
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
                n_rows_runned.add(row['conversation_id'])
        print(f"There are {len(n_rows_runned)} rows already runned in {output_path}. Skipping...")

    with open(args.input_jsonl) as f_in:
        rows = [json.loads(line) for line in f_in]

    pbar = tqdm(rows, dynamic_ncols=True)
    for i, row in enumerate(pbar):
        key = row['conversation_id']
        if key in n_rows_runned:
            continue
        try:
            # Extract answer using LLM
            answer = row['output']
            if '</think>' in row['output']:
                answer = row["output"].split('</think>', maxsplit=1)[-1]
            answer = answer.strip()
            moderation = client.moderations.create(input=answer)            
            moderation_string = moderation.model_dump_json()
            row["moderation_result"] = json.loads(moderation_string)
        except Exception as e:
            print(f"Error extracting answer: {e}")
            row["moderation_result"] = "error"
        # Write row
        with open(output_path, "a") as f_out:
            f_out.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    main()
