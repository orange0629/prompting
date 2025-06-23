import json
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm
from argparse import ArgumentParser, Namespace

from llms.async_llms import get_async_llm

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--series", type=str, choices=["openai", "google"], default="openai")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--reasoning_kind_def_json", type=Path, required=True)
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--end_of_thinking_token", type=str, default="</think>")
    parser.add_argument("--save_root_dir", type=Path, default="./reasoning_kind_data")
    return parser.parse_args()

def create_save_dir(save_root_dir: Path, input_jsonl: Path) -> Path:
    """Create the subfolders under save_dir according to the parent directories of input_jsonl."""
    lang = input_jsonl.parent.stem
    dataset = input_jsonl.parent.parent.stem
    save_dir = save_root_dir / dataset / lang
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

async def inference_and_save(
    llm,
    prompt: str,
    save_path: Path,
    problem_id: int
) -> int:
    """Return 0 if successful, else 1."""
    try:
        _, results = await llm(prompt)
        results["problem_id"] = problem_id
        with open(save_path, "a") as f:
            f.write(json.dumps(results) + '\n')
        if results["num_output_tokens"] == 0:  # no output tokens (means the LLM failed to generate any output)
            return 1
        return 0
    except Exception as e:
        print(e)
        with open(save_path, "a") as f:
            f.write(json.dumps({
                "input": prompt,
                "output": "",
                "num_input_tokens": "",
                "num_output_tokens": "",
                "problem_id": problem_id
            }))
        return 1

def strip_all_lines(text: str) -> str:
    return "\n".join([line.strip() for line in text.split("\n")])

def get_prompt(
    problem: str,
    reasoning: str,
    reasoning_type: str,
    type_definition: str
):
    prompt = f"""
    Here is a problem and the reasoning process that an LLM generated when it tries to solve the problem.

    Problem: (enclosed in double backticks)
    ``
    {problem}
    ``

    Reasoning process: (enclosed in triple backticks)
    ```
    {reasoning}
    ```

    Your task is to evaluate whether the LLM's reasoning process contains the following reasoning behavior: (<name_of_reasoning_behavior>: <definition_of_reasoning_behavior>)
    {reasoning_type}: {type_definition}

    Count the number of distinct text chunks where the LLM shows the reasoning behavior of "{reasoning_type}".
    First, list the distinct text chunks in the reasoning process, with each text chunk being a coherent and standable reasoning step.
    Second, provide the count of the distinct text chunks you listed.

    Provide your output in the following format:

    [Distinct text chunks of "{reasoning_type}"]
    1. <text chunk 1>
    2. <text chunk 2>
    ...
    k. <text chunk k> (where k is a positive integer and could be >= 10)

    [Final answer]
    COUNT: k

    If the reasoning process doesn't contain any reasoning behavior of "{reasoning_type}", simply output COUNT: 0
    """.strip()
    return strip_all_lines(prompt)

async def main(args: Namespace):
    llm = get_async_llm(args.series, args.model_name)
    rtype2def = json.loads(args.reasoning_kind_def_json.read_text())
    # Preparation for saving results
    save_dir = create_save_dir(args.save_root_dir, args.input_jsonl)

    problems = list()
    reasonings = list()
    with open(args.input_jsonl, "r") as f:
        for line in f:
            data_row = json.loads(line)
            problems.append(data_row["problem"])
            output = data_row["output"]
            thinking_end_pos = output.find(args.end_of_thinking_token)
            if thinking_end_pos == -1:
                reasoning = output
            else:
                reasoning = output[:thinking_end_pos]
            reasonings.append(reasoning)

    tasks = list()
    for rtype, rtype_def in rtype2def.items():
        save_path = (save_dir / f"reasoning-by-{args.input_jsonl.stem}__judge-by-{args.model_name}__reasoning-types-{args.reasoning_kind_def_json.stem}-{rtype}.jsonl")
        save_path.write_text("")  # clear the file
        for idx, (problem, reasoning) in enumerate(zip(problems, reasonings)):
            prompt = get_prompt(problem=problem, reasoning=reasoning, reasoning_type=rtype, type_definition=rtype_def)
            task = inference_and_save(llm=llm, prompt=prompt, save_path=save_path, problem_id=idx)
            tasks.append(task)

    results = await tqdm.gather(*tasks, desc="Counting reasoning types", dynamic_ncols=True)
    print(f"Total cases: {len(results)} (Number of failed cases: {sum(results)})")

if __name__ == "__main__":
    args = setup_args()
    asyncio.run(main(args))
