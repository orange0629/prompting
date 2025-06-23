import time
import json
from dotenv import load_dotenv
from pathlib import Path
from argparse import ArgumentParser, Namespace

from count_reasoning_types import create_save_dir
from count_reasoning_types_batch import upload_to_gcs, create_batch_job, find_predictions_jsonl, download_from_gcs

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--series", type=str, choices=["openai", "google"], default="google")
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash-001")
    parser.add_argument("--reasoning_kind_def_json", type=Path, required=True)
    parser.add_argument("--input_jsonl", type=Path, required=True, help="The jsonl file containing the segmented reasoning steps of the LLM")
    parser.add_argument("--batch_prediction_jsonl_dir", type=Path, default="./batch_prediction_input")
    parser.add_argument("--gcs_upload_prefix", type=str, default="gs://exp3-gemini-batch-input/reasoning_kind_classification/batch_prediction_input")
    parser.add_argument("--gcs_download_prefix", type=str, default="gs://exp3-gemini-batch-input/reasoning_kind_classification/batch_prediction_output")
    parser.add_argument("--batch_prediction_output_dir", type=Path, default="./batch_prediction_output")
    parser.add_argument("--end_of_thinking_token", type=str, default="</think>")
    parser.add_argument("--save_root_dir", type=Path, default="./reasoning_kind_classification")
    return parser.parse_args()

def extract_path_components(input_jsonl: Path) -> tuple[str, str, str]:
    """Extract dataset name, language, and filename from the input path."""
    path_parts = str(input_jsonl).split('/')
    dataset_name = path_parts[1]
    language = path_parts[2]
    filename = path_parts[-1]
    # assert (dataset_name == "MATH-500") and (language in {"en", "es", "ja", "ko", "ru", "sw", "te", "zh-CN"})
    return dataset_name, language, filename

def get_file_save_name(input_jsonl: Path) -> str:
    """Create a save filename from the input path components."""
    dataset_name, language, filename = extract_path_components(input_jsonl)
    return '__'.join([dataset_name, language, filename])

def get_gcs_upload_path(input_jsonl: Path, gcs_upload_prefix: str) -> str:
    """Generate the GCS upload path for the input file."""
    dataset_name, language, filename = extract_path_components(input_jsonl)
    suffix = '/'.join([dataset_name, language, filename])
    return f"{gcs_upload_prefix}/{suffix}"

def get_gcs_download_dir(input_jsonl: Path, gcs_download_prefix: str) -> str:
    """Generate the GCS download directory for the input file."""
    dataset_name, language, filename = extract_path_components(input_jsonl)
    suffix = '/'.join([dataset_name, language, filename])
    return f"{gcs_download_prefix}/{suffix}"

def strip_all_lines(text: str) -> str:
    return "\n".join([line.strip() for line in text.split("\n")])

def get_rtype2def_str(rtype2def: dict[str, str]) -> str:
    rtype2def["Others"] = "This reasoning step is the continuation of the previous reasoning step, or it does not fall into any of the above categories."
    return "\n".join([f"{idx + 1}. {name}: {definition}" for idx, (name, definition) in enumerate(rtype2def.items())])

def get_classification_prompt(
    problem: str,
    reasoning: str,
    rtype2def_str: str
) -> str:
    prompt = f"""
    Here is a problem and the reasoning process that an LLM generated when it tries to solve the problem.

    Problem: (enclosed in double backticks)
    ``
    {problem}
    ``

    Reasoning process: (enclosed in triple backticks, the reasoning process has been split into distinct reasoning steps in the format of <step_idx><reasoning_step_content></step_idx>)
    ```
    {reasoning}
    ```

    Your task is to classify each reasoning step into one of the following reasoning types: (specified by <type_index>. <type_name>: <definition>)
    {rtype2def_str}

    Generate the rationale before you make the classification.
    Provide your output in the following format:

    [Reasoning]
    <step_1><rationale_1><type_name_1></step_1>
    <step_2><rationale_2><type_name_2></step_2>
    ...

    [Final answer]
    <step_1><type_name_1></step_1>
    <step_2><type_name_2></step_2>
    ...
    """.strip()
    return strip_all_lines(prompt)

def format_reasoning_steps(reasoning_steps: list[str]) -> str:
    """Curate the steps in the format of <step_idx><reasoning_step_content></step_idx>"""
    return '\n'.join([f"<step_{idx}>{step}</step_{idx}>" for idx, step in enumerate(reasoning_steps, start=1)])

def create_batch_prediction_input_jsonl(
    input_jsonl: Path,
    rtype2def_str: str,
    batch_prediction_jsonl_dir: Path,
) -> Path:
    batch_prediction_jsonl_dir.mkdir(parents=True, exist_ok=True)
    batch_prediction_jsonl_path = batch_prediction_jsonl_dir / get_file_save_name(input_jsonl=input_jsonl)
    batch_prediction_jsonl_path.write_text('')
    with open(input_jsonl, 'r') as f:
        for i, line in enumerate(f):
            data_row = json.loads(line)
            reasoning_steps = data_row["formatted_steps"]
            reasoning = format_reasoning_steps(reasoning_steps)
            prompt = get_classification_prompt(
                problem=data_row["input"],
                reasoning=reasoning,
                rtype2def_str=rtype2def_str
            )
            with open(batch_prediction_jsonl_path, 'a') as f:
                f.write(json.dumps({
                    "id": str(i),
                    "request": {"contents": [{"parts": {"text": prompt}, "role": "user"}]}
                }) + '\n')
    return batch_prediction_jsonl_path

def main(args: Namespace):
    rtype2def = json.loads(args.reasoning_kind_def_json.read_text())
    rtype2def_str = get_rtype2def_str(rtype2def)
    jsonl_for_batch_upload = create_batch_prediction_input_jsonl(
        input_jsonl=args.input_jsonl,
        rtype2def_str=rtype2def_str,
        batch_prediction_jsonl_dir=args.batch_prediction_jsonl_dir
    )
    gcs_upload_path = get_gcs_upload_path(input_jsonl=args.input_jsonl, gcs_upload_prefix=args.gcs_upload_prefix)
    gcs_jsonl_success = upload_to_gcs(jsonl_for_batch_upload, gcs_upload_path)
    if not gcs_jsonl_success:
        raise NotImplementedError

    gcs_download_dir = get_gcs_download_dir(input_jsonl=args.input_jsonl, gcs_download_prefix=args.gcs_download_prefix)
    batch_job = create_batch_job(
        model_name=args.model_name,
        input_uri=gcs_upload_path,
        output_uri=gcs_download_dir
    )

    jsonl_path_for_download = find_predictions_jsonl(output_uri=gcs_download_dir)
    if jsonl_path_for_download:
        print(f"Found predictions.jsonl at: {jsonl_path_for_download}")
    else:
        raise ValueError("Could not find predictions.jsonl in the specified location")

    batch_prediction_output_path = args.batch_prediction_output_dir / get_file_save_name(input_jsonl=args.input_jsonl)
    download_success = download_from_gcs(jsonl_path_for_download=jsonl_path_for_download, batch_prediction_output_path=batch_prediction_output_path)
    if not download_success:
        raise ValueError("Could not download predictions.jsonl from GCS")

    # Save to the jsonl files corresponding to each reasoning type
    save_dir = create_save_dir(save_root_dir=args.save_root_dir, input_jsonl=args.input_jsonl)
    save_path = save_dir / f"reasoning-by-{args.input_jsonl.stem}__judge-by-{args.model_name}__reasoning-types-{args.reasoning_kind_def_json.stem}.jsonl"
    save_path.write_text('')
    with open(batch_prediction_output_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            id_ = data["id"]
            problem_id = id_.split('-')[0]
            with open(save_path, 'a') as f:
                f.write(json.dumps({
                    "input": data["request"]["contents"][0]["parts"]["text"],
                    "output": data["response"]["candidates"][0]["content"]["parts"][0]["text"],
                    "num_input_tokens": data["response"]["usageMetadata"]["promptTokenCount"],
                    "num_output_tokens": data["response"]["usageMetadata"]["candidatesTokenCount"],
                    "problem_id": problem_id
                }) + '\n')

if __name__ == "__main__":
    load_dotenv()
    args = setup_args()

    start_time = time.time()
    main(args)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
