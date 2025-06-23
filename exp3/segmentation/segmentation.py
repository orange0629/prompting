import json
import torch
import logging
import argparse
import os
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List

import warnings
warnings.filterwarnings("ignore")

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def predict_batch_step_splits(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    sep_token: str,
    device: torch.device
) -> List[List[int]]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192
    ).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)  # [B, T]

    results = []
    sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)

    for i in range(len(texts)):
        input_ids = inputs["input_ids"][i]
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
        pred = [predictions[i, pos].item() for pos in sep_positions]
        results.append(pred)

    return results

def format_steps(text: str, split_predictions: List[int], sep_token: str = "[SEP]") -> List[str]:
    text_parts = text.split(sep_token)
    formatted_steps = []
    current_step = text_parts[0]

    for i, pred in enumerate(split_predictions):
        if i + 1 < len(text_parts):
            if pred == 1:
                formatted_steps.append(current_step.strip())
                current_step = text_parts[i + 1]
            else:
                current_step += " " + text_parts[i + 1] + '\n'

    if current_step.strip():
        formatted_steps.append(current_step.strip())

    return formatted_steps

def main():
    parser = argparse.ArgumentParser(description="Use a trained model to predict step splits (batched)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str)
    parser.add_argument("--output_jsonl", type=Path)
    parser.add_argument("--sep_token", type=str, default="[SEP]")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger.info(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path).to(device)
    model.eval()

    with open(args.input_jsonl, 'r') as f:
        lines = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Processing {len(lines)} examples from {args.input_jsonl}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for i in tqdm(range(0, len(lines), args.batch_size), dynamic_ncols=True):
        batch = lines[i:i + args.batch_size]
        raw_texts = [x["model_output"].replace("\n\n", args.sep_token).replace("\n", args.sep_token) for x in batch]

        try:
            batch_preds = predict_batch_step_splits(raw_texts, model, tokenizer, args.sep_token, device)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"⚠️  Skipping batch {i} due to OOM")
            torch.cuda.empty_cache()
            continue

        for ex, pred, reasoning_text in zip(batch, batch_preds, raw_texts):
            steps = format_steps(reasoning_text, pred, args.sep_token)
            results.append({
                "input": ex["full_input_prompt"],
                "reasoning": reasoning_text,
                "formatted_steps": steps,
                "num_steps": len(steps)
            })

        if args.debug:
            break

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("✅ Done.")

if __name__ == "__main__":
    main()
