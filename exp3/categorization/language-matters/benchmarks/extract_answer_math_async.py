import os
import re
import json
import asyncio
import hashlib
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from colorama import Fore, Style
from argparse import ArgumentParser, Namespace

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--think_token_eos", type=str, default=None)
    parser.add_argument("--think_token_bos", type=str, default=None)
    parser.add_argument("--use_last_line", action="store_true")
    parser.add_argument("--answer_jsonl", type=Path, default="log/MATH-500/en/DeepSeek-R1-Distill-Llama-70B.jsonl")
    return parser.parse_args()

EXTRACT_ANSWER_TEMPLATE = """
The following text is an LLM's response to a math question:

Text (enclosed in triple backticks):
```
{text}
```

Extract the LLM's answer from the text. Your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the math question.
If there is no answer in the text, you may respond with Answer: None.
""".strip()

EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

client = AsyncOpenAI()

async def chat_completion(prompt: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.0
    )
    return response.choices[0].message.content

async def extract_answer(text: str, thinking_end_token: str | None, use_last_line: bool = False) -> str:
    if ' <sep> human:' in text: # handle reka model
        text = text.replace(' <sep> human:','')

    if thinking_end_token is None:
        answer_text = text
    else:
        thinking_and_answer = text.split(thinking_end_token)
        if len(thinking_and_answer) < 2:
            if use_last_line:
                answer_text = '\n'.join(text.split('\n')[-2:])
            else:
                print(f"No {thinking_end_token} found in the text")
                return f"No {thinking_end_token} found in the text"
        else:
            answer_text = thinking_and_answer[-1].strip()
    prompt = EXTRACT_ANSWER_TEMPLATE.format(text=answer_text)
    response = await chat_completion(prompt)
    match = re.search(r"(?i)Answer\s*:\s*([^\n]+)", response)
    extracted_answer = match.group(1).strip() if match else "None"
    return extracted_answer

async def check_equality(prediction: str, ground_truth: str) -> bool:
    """True: prediction is correct, False: prediction is incorrect"""
    prompt = EQUALITY_TEMPLATE % {
        "expression1": prediction,
        "expression2": ground_truth
    }
    response = await chat_completion(prompt)
    return response.strip() == "Yes"

def is_reasoning_model(model_name: str) -> bool:
    if model_name.lower().startswith("deepseek-r1"):
        return True
    return False

def read_existing_results(file_path: Path) -> list:
    """Read existing results from a file if it exists"""
    existing_results = []
    if file_path.exists():
        print(Fore.CYAN + f"Reading existing results from {file_path}" + Style.RESET_ALL)
        with open(file_path, "r") as f_in:
            for line in f_in:
                try:
                    existing_results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line in existing file: {e}")
        print(f"Found {len(existing_results)} existing results.")
    return existing_results

async def main(args: Namespace) -> None:
    if not os.path.exists(args.answer_jsonl):
        args.answer_jsonl = args.input_jsonl
        print(Fore.YELLOW + f"No answer file provided, using input file as answer file: {args.answer_jsonl}" + Style.RESET_ALL)
    # Setup output path first to check for existing results
    output_dir = args.input_jsonl.parent / "answer_extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.input_jsonl.name

    # Check if output file already exists and read existing results
    existing_results = read_existing_results(output_path)
    existing_data_map = {}
    if existing_results:
        print(Fore.YELLOW + f"Found existing results in {output_path}. Will only process new inputs." + Style.RESET_ALL)
        # Create a map of processed outputs to avoid reprocessing
        for i, result in enumerate(existing_results):
            if 'output_hash' in result:
                existing_data_map[result['output_hash']] = i
            else:
                # For backwards compatibility with files without hashes
                print(Fore.YELLOW + "Warning: Existing results don't have output hashes. Will use them as-is." + Style.RESET_ALL)

    # Extract the outputs and ground truth answers from the JSONL file
    print(f"Extracting answers from {args.input_jsonl}")
    all_input_data = []
    outputs_to_process = []
    ground_truth_answers_to_process = []
    skipped_count = 0
    
    with open(args.input_jsonl) as f_in, open(args.answer_jsonl) as f_ans:
        for line_index, line in enumerate(f_in):
            data = json.loads(line)
            line_ans = f_ans.readline()
            data_ans = json.loads(line_ans)
            output = data["output"]
            answer = data_ans["answer"]
            
            # Store all input data for reference
            all_input_data.append(data)
            
            # Create a deterministic hash of the output to use as a unique identifier
            import hashlib
            output_hash = hashlib.md5(output.encode('utf-8')).hexdigest()
            
            # Check if this output has already been processed
            if output_hash in existing_data_map:
                skipped_count += 1
                continue
                
            # If not already processed, add to the lists for processing
            outputs_to_process.append(output)
            ground_truth_answers_to_process.append(answer)
    
    if skipped_count > 0:
        print(Fore.GREEN + f"Skipping {skipped_count} already processed inputs." + Style.RESET_ALL)
    
    if not outputs_to_process:
        print(Fore.GREEN + "All inputs have already been processed. Nothing new to do." + Style.RESET_ALL)
        return
    
    print(Fore.CYAN + f"Processing {len(outputs_to_process)} new inputs." + Style.RESET_ALL)
    
    # Extract the answers from the outputs
    if is_reasoning_model(args.input_jsonl.stem):
        thinking_end_token = "</think>"
    else:
        thinking_end_token = args.think_token_eos
    print(Fore.CYAN + f"Using thinking end token {thinking_end_token}" + Style.RESET_ALL)
    extract_tasks = [extract_answer(output, thinking_end_token, args.use_last_line) for output in outputs_to_process]
    extracted_answers = await tqdm.gather(*extract_tasks, desc="Extracting answers")

    # Check the equality of the extracted answers and the ground truth answers
    equality_tasks = [check_equality(extracted_answer, ground_truth_answer) 
                     for extracted_answer, ground_truth_answer in zip(extracted_answers, ground_truth_answers_to_process)]
    equalities = await tqdm.gather(*equality_tasks, desc="Checking equalities")
    
    if equalities:
        print(f"Accuracy of new inputs: {sum(equalities) / len(equalities)}")
    
    # Prepare new results
    new_results = []
    for i, (extracted_answer, equality) in enumerate(zip(extracted_answers, equalities)):
        import hashlib
        output_hash = hashlib.md5(outputs_to_process[i].encode('utf-8')).hexdigest()
        new_results.append({
            "answer_extracted": extracted_answer, 
            "equality": equality,
            "output_hash": output_hash
        })
    
    # Combine existing and new results
    all_results = existing_results + new_results
    
    # Save all results
    print(Fore.CYAN + f"Saving results to {output_path}" + Style.RESET_ALL)
    with open(output_path, "w") as f_out:
        for result in all_results:
            f_out.write(json.dumps(result) + "\n")
    
    if existing_results:
        print(f"Combined {len(existing_results)} existing results with {len(new_results)} new results.")
    else:
        print(f"Saved {len(new_results)} new results.")

if __name__ == "__main__":
    args = setup_args()
    asyncio.run(main(args))