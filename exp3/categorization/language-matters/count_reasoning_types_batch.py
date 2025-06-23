import time
import json
from dotenv import load_dotenv
from pathlib import Path
from argparse import ArgumentParser, Namespace
from google import genai
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions

from count_reasoning_types import get_prompt, create_save_dir

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--series", type=str, choices=["openai", "google"], default="google")
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash-001")
    parser.add_argument("--reasoning_kind_def_json", type=Path, required=True)
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--batch_prediction_jsonl_dir", type=Path, default="./batch_prediction_input")
    parser.add_argument("--gcs_upload_prefix", type=str, default="gs://appier-ai-research-dev/multilingual-reasoning/cotscope/reasoning_kind_data/batch_prediction_input")
    parser.add_argument("--gcs_download_prefix", type=str, default="gs://appier-ai-research-dev/multilingual-reasoning/cotscope/reasoning_kind_data/batch_prediction_output")
    parser.add_argument("--batch_prediction_output_dir", type=Path, default="./batch_prediction_output")
    parser.add_argument("--end_of_thinking_token", type=str, default="</think>")
    parser.add_argument("--save_root_dir", type=Path, default="./reasoning_kind_data")
    return parser.parse_args()

def get_file_save_name(input_jsonl: Path) -> str:
    return '__'.join(str(input_jsonl).split('/')[-3:])

def get_gcs_upload_path(input_jsonl: Path, gcs_upload_prefix: str) -> str:
    suffix = '/'.join(str(input_jsonl).split('/')[-3:])
    return gcs_upload_prefix + '/' + suffix

def get_gcs_download_dir(input_jsonl: Path, gcs_download_prefix: str) -> str:
    suffix = '/'.join(str(input_jsonl).split('/')[-3:])
    return gcs_download_prefix + '/' + suffix

def create_batch_job(
    model_name: str,
    input_uri: str,
    output_uri: str,
) -> genai.types.BatchJob:
    client = genai.Client(http_options=HttpOptions(api_version="v1"),
                               vertexai=True,
      project="gemini-batch-classification",
      location="us-central1",
      )
    # See the documentation: https://googleapis.github.io/python-genai/genai.html#genai.batches.Batches.create
    job = client.batches.create(
        model=model_name,
        # Source link: https://storage.cloud.google.com/cloud-samples-data/batch/prompt_for_batch_gemini_predict.jsonl
        src=input_uri,
        config=CreateBatchJobConfig(dest=output_uri),
    )
    print(f"Job name: {job.name}")
    print(f"Job state: {job.state}")
    # Example response:
    # Job name: projects/%PROJECT_ID%/locations/us-central1/batchPredictionJobs/9876453210000000000
    # Job state: JOB_STATE_PENDING

    # See the documentation: https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob
    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED,
    }

    while job.state not in completed_states:
        time.sleep(30)
        job = client.batches.get(name=job.name)
        print(f"Job state: {job.state}")
    return job

def find_predictions_jsonl(output_uri: str) -> str | None:
    """
    Recursively search for predictions.jsonl in the specified GCS bucket and prefix.
    
    Args:
        bucket_name: Name of the GCS bucket
        prefix: Prefix (folder path) to search in
        
    Returns:
        Full GCS path to predictions.jsonl if found, None otherwise
    """
    bucket_name = output_uri.replace('gs://', '').split('/')[0]
    prefix = '/'.join(output_uri.replace('gs://', '').split('/')[1:])

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    
    # List all blobs with the given prefix
    blobs = bucket.list_blobs(prefix=prefix)
    
    jsonl_path = None
    latest_blob = None
    latest_time = None
    
    for blob in blobs:
        if blob.name.endswith('predictions.jsonl'):
            if latest_time is None or blob.updated > latest_time:
                latest_blob = blob
                latest_time = blob.updated
                jsonl_path = f"gs://{bucket_name}/{blob.name}"
    
    if jsonl_path is None:
        raise ValueError("Could not find predictions.jsonl in the specified location")
    
    if latest_blob:
        print(f"Using latest predictions.jsonl updated at {latest_time}")
    return jsonl_path

def create_batch_prediction_input_jsonl(
    input_jsonl: Path,
    rtype2def: dict[str, str],
    batch_prediction_jsonl_dir: Path,
) -> Path:
    """Load the rows from input_jsonl, prepare prompts, and make the local jsonl.

    Format: {"id": 1, "request": {"contents": [{"parts": {"text": "Give me a recipe for banana bread."}, "role": "user"}]}}
    """
    batch_prediction_jsonl_dir.mkdir(parents=True, exist_ok=True)
    batch_prediction_jsonl_path = batch_prediction_jsonl_dir / get_file_save_name(input_jsonl=input_jsonl)
    batch_prediction_jsonl_path.write_text('')
    with open(input_jsonl, 'r') as f:
        for i, line in enumerate(f):
            data_row = json.loads(line)
            output = data_row["output"]
            thinking_end_pos = output.find(args.end_of_thinking_token)
            if thinking_end_pos == -1:
                reasoning = output
            else:
                reasoning = output[:thinking_end_pos]
            for rtype, rtype_def in rtype2def.items():
                prompt = get_prompt(
                    problem=data_row["problem"],
                    reasoning=reasoning,
                    reasoning_type=rtype,
                    type_definition=rtype_def
                )
                with open(batch_prediction_jsonl_path, 'a') as f:
                    f.write(json.dumps({
                        "id": f"{i}-{rtype}",
                        "request": {"contents": [{"parts": {"text": prompt}, "role": "user"}]}
                    }) + '\n')
    return batch_prediction_jsonl_path

def upload_to_gcs(jsonl_for_batch_upload: Path, gcs_upload_path: str) -> bool:
    """
    Upload a local jsonl file to Google Cloud Storage.
    
    Args:
        jsonl_for_batch_upload: Path to the local jsonl file
        gcs_upload_path: GCS path where the file should be uploaded
        
    Returns:
        bool: True if upload was successful, False otherwise
    """
    try:
        from google.cloud import storage
        
        # Initialize the GCS client
        storage_client = storage.Client()
        
        # Parse the GCS path to get bucket name and blob path
        if not gcs_upload_path.startswith("gs://"):
            print(f"Error: GCS path must start with 'gs://'. Got: {gcs_upload_path}")
            return False
            
        path_parts = gcs_upload_path[5:].split('/', 1)
        if len(path_parts) != 2:
            print(f"Error: Invalid GCS path format: {gcs_upload_path}")
            return False
            
        bucket_name, blob_path = path_parts
        
        # Get the bucket
        bucket = storage_client.bucket(bucket_name)
        
        # Create a blob and upload the file
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(jsonl_for_batch_upload, timeout=600)
        
        print(f"Successfully uploaded {jsonl_for_batch_upload} to {gcs_upload_path}")
        return True
        
    except ImportError:
        print("Error: google-cloud-storage package is not installed. Please install it with 'pip install google-cloud-storage'")
        return False
    except Exception as e:
        print(f"Error uploading to GCS: {str(e)}")
        return False

def download_from_gcs(jsonl_path_for_download: str, batch_prediction_output_path: Path) -> bool:
    """Download the predictions.jsonl from GCS and save to the batch_prediction_output_path (which is a jsonl file)."""
    try:
        from google.cloud import storage
        
        # Initialize the GCS client
        storage_client = storage.Client()
        
        # Parse the GCS path to get bucket name and blob path
        if not jsonl_path_for_download.startswith("gs://"):
            print(f"Error: GCS path must start with 'gs://'. Got: {jsonl_path_for_download}")
            return
            
        path_parts = jsonl_path_for_download[5:].split('/', 1)
        if len(path_parts) != 2:
            print(f"Error: Invalid GCS path format: {jsonl_path_for_download}")
            return
            
        bucket_name, blob_path = path_parts
        
        # Make sure the output directory exists
        batch_prediction_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get the bucket and blob
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Download the file
        blob.download_to_filename(batch_prediction_output_path)
        
        print(f"Successfully downloaded {jsonl_path_for_download} to {batch_prediction_output_path}")
        return True
    except ImportError:
        print("Error: google-cloud-storage package is not installed. Please install it with 'pip install google-cloud-storage'")
    except Exception as e:
        print(f"Error downloading from GCS: {str(e)}")
    return False

def main(args: Namespace) -> None:
    rtype2def = json.loads(args.reasoning_kind_def_json.read_text())
    jsonl_for_batch_upload = create_batch_prediction_input_jsonl(
        input_jsonl=args.input_jsonl,
        rtype2def=rtype2def,
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
    rtype2save_path = {
        rtype: save_dir / f"reasoning-by-{args.input_jsonl.stem}__judge-by-{args.model_name}__reasoning-types-{args.reasoning_kind_def_json.stem}-{rtype}.jsonl"
        for rtype in rtype2def.keys()
    }
    for save_path in rtype2save_path.values():
        save_path.write_text('')
    with open(batch_prediction_output_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            id_ = data["id"]
            problem_id = id_.split('-')[0]
            reasoning_type = '-'.join(id_.split('-')[1:])
            save_path = rtype2save_path[reasoning_type]
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
