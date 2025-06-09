#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Four-metric reward model with margin-weighted pairwise loss
—————————————————————————————————————————————————————————————
Metrics:
    • acc_mean
    • acc_var
    • output_tokens_var
    • consistency
Each JSONL line must contain:
    {
        "benchmark": "...",
        "model": "...",
        "prompt_id": "...",
        "acc_mean": 0.27,
        "acc_var": 0.19,
        "output_tokens_var": 1.3e6,
        "consistency": 0.03,
        "prompt": "... full prompt ..."
    }
"""

# ───────────────────────────────── imports ──────────────────────────────────
import os
import sys
sys.path.append(os.path.abspath("../"))
import argparse
import json
import random
import math
from itertools import combinations
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
)

from trl import RewardConfig
from trl.trainer.reward_trainer import RewardTrainer
from tqdm import tqdm

# Custom Import
from lib.utils import MinMaxNormalizer

# ───────────────────────── global configuration ────────────────────────────
METRIC_COLS: List[str] = [
    "acc_mean",
]
NUM_METRICS: int = len(METRIC_COLS)
RANDOM_SEED: int = 42

# ───────────────────────────── utilities ────────────────────────────────────
def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ─────────────── build pairwise comparison dataset ──────────────────────────
def build_pairs(
    rows: List[Dict[str, Any]],
    prompt_template: str,
    normalizer: MinMaxNormalizer,
    max_pairs: int = 100_000,
) -> Dataset:
    pairs = list(combinations(rows, 2))
    items = []
    for a, b in pairs:
        items.append(
            {
                "example_a": prompt_template.format(
                    system_prompt=a["prompt"].replace(" /// ", " ")
                ),
                "example_b": prompt_template.format(
                    system_prompt=b["prompt"].replace(" /// ", " ")
                ),
                "scores_a": [normalizer.normalize(m, a[m]) for m in METRIC_COLS],
                "scores_b": [normalizer.normalize(m, b[m]) for m in METRIC_COLS],
            }
        )
    random.shuffle(items)
    return Dataset.from_list(items)

# ──────────────────────── data collator ─────────────────────────────────────
class RewardDataCollatorWithPadding:
    """Pads A/B sequences and stacks margin tensor (batch, 4)."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        padding: Union[bool, str] = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, batch: List[Dict[str, Any]]):
        a, b, margins = [], [], []
        for ex in batch:
            a.append(
                {"input_ids": ex["input_ids_a"], "attention_mask": ex["attention_mask_a"]}
            )
            b.append(
                {"input_ids": ex["input_ids_b"], "attention_mask": ex["attention_mask_b"]}
            )
            margins.append(ex["margin"])

        batch_a = self.tokenizer.pad(
            a,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_b = self.tokenizer.pad(
            b,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        return {
            "input_ids_a": batch_a["input_ids"],
            "attention_mask_a": batch_a["attention_mask"],
            "input_ids_b": batch_b["input_ids"],
            "attention_mask_b": batch_b["attention_mask"],
            "margin": torch.tensor(margins, dtype=torch.float),
            "return_loss": True,
        }

# ─────────────── custom trainer with margin-weighted loss ───────────────────
# class PromptRewardTrainer(RewardTrainer):
#     def _tokenize(self, batch):
#         # dataset is already tokenised → return as-is
#         return batch
    
#     def compute_loss(self, model, inputs, return_outputs: bool = False):
#         r_a = model(
#             input_ids=inputs["input_ids_a"], attention_mask=inputs["attention_mask_a"]
#         ).logits  # (B,4)
#         r_b = model(
#             input_ids=inputs["input_ids_b"], attention_mask=inputs["attention_mask_b"]
#         ).logits  # (B,4)

#         margin = inputs["margin"].to(r_a.device)  # (B,4)
#         sign = torch.sign(margin)
#         # margin-weighted pairwise loss
#         loss_per_metric = -torch.nn.functional.logsigmoid(sign * (r_a - r_b)) * margin.abs()
#         loss = loss_per_metric.mean()

#         if return_outputs:
#             return loss, {"rewards_a": r_a, "rewards_b": r_b}
#         return loss

class PairwiseTrainer(Trainer):
    """
    Trainer with margin-weighted, multi-metric pairwise loss.
    * Expects the batch keys produced by RewardDataCollatorWithPadding *
        input_ids_a / attention_mask_a
        input_ids_b / attention_mask_b
        margin  ->  tensor (B, 4)
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r_a = model(
            input_ids=inputs["input_ids_a"],
            attention_mask=inputs["attention_mask_a"],
        ).logits                       # shape (B, 4)
        r_b = model(
            input_ids=inputs["input_ids_b"],
            attention_mask=inputs["attention_mask_b"],
        ).logits

        margin = inputs["margin"].to(r_a.device)       # (B, 4)
        sign   = torch.sign(margin)

        # margin-weighted pairwise loss
        per_metric = -torch.nn.functional.logsigmoid(sign * (r_a - r_b)) * margin.abs()
        loss = per_metric.mean()

        if return_outputs:
            return loss, {"rewards_a": r_a, "rewards_b": r_b}
        return loss
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        return (loss.detach(), logits_dict["rewards_a"] - logits_dict["rewards_b"], inputs["margin"])


# ─────────────────────────────── metrics ────────────────────────────────────
def compute_accuracy(eval_preds):
    predictions, labels = eval_preds  # predictions.shape = (N, num_metrics)
    predictions = predictions.astype(np.float32)
    labels = labels.astype(np.float32)

    acc_all = (np.sign(predictions) == np.sign(labels))
    overall_acc = acc_all.mean()

    metrics = {"overall_acc": overall_acc}
    for i, m in enumerate(METRIC_COLS):
        metrics[f"{m}_acc"] = acc_all[:, i].mean()

    return metrics

# ─────────────── tokenisation helper (returns dict) ─────────────────────────
def make_tok_pair_fn(tokenizer: PreTrainedTokenizerBase, max_len: int):
    def _fn(example):
        tok_a = tokenizer(
            example["example_a"], truncation=True, max_length=max_len
        )
        tok_b = tokenizer(
            example["example_b"], truncation=True, max_length=max_len
        )
        margin = (
            np.array(example["scores_a"], dtype=np.float32)
            - np.array(example["scores_b"], dtype=np.float32)
        ) * 100.0
        return {
            "input_ids_a": tok_a["input_ids"],
            "attention_mask_a": tok_a["attention_mask"],
            "input_ids_b": tok_b["input_ids"],
            "attention_mask_b": tok_b["attention_mask"],
            "margin": margin.tolist(),
        }

    return _fn

# ─────────────────────── training routine ───────────────────────────────────
def train_model(args):
    set_seed()

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint,
        num_labels=NUM_METRICS,
        torch_dtype=torch.bfloat16,
        #use_flash_attention_2=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    rows = load_jsonl(args.train_path)
    train_rows, dev_rows = train_test_split(rows, test_size=0.4, random_state=RANDOM_SEED)
    dev_rows, test_rows = train_test_split(dev_rows, test_size=0.5, random_state=RANDOM_SEED)

    normalizer = MinMaxNormalizer.from_data(train_rows, METRIC_COLS)
    os.makedirs(args.output_dir, exist_ok=True)
    normalizer.save(os.path.join(args.output_dir, "normalizer.json"))

    prompt_template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        if "llama" in args.model_checkpoint.lower()
        else "{system_prompt}"
    )

    train_ds = build_pairs(train_rows, prompt_template, normalizer).select(
        range(len(train_rows)*int(math.sqrt(len(train_rows))))
    )
    dev_ds = build_pairs(dev_rows, prompt_template, normalizer).select(
        range(len(dev_rows)*int(math.sqrt(len(dev_rows))))
    )

    train_ds = train_ds.map(
        make_tok_pair_fn(tokenizer, args.max_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    dev_ds = dev_ds.map(
        make_tok_pair_fn(tokenizer, args.max_length),
        batched=True,
        remove_columns=dev_ds.column_names,
    )

    
    # ─────────────────────────── 3. training args ──────────────────────
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="overall_acc",
        greater_is_better=True,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        logging_strategy="steps",
        logging_steps=1,
        report_to="wandb",          # flip to "wandb" or "tensorboard" if desired
        remove_unused_columns=False,
    )

    # ─────────────────────────── 4. initialise trainer ────────────────
    trainer = PairwiseTrainer(
        model=model,
        args=train_args,
        tokenizer=tokenizer,                               # required for saving / gen
        train_dataset=train_ds,                            # already tokenised
        eval_dataset=dev_ds,
        data_collator=RewardDataCollatorWithPadding(tokenizer),
        compute_metrics=compute_accuracy,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    

# ─────────────────────────── testing routine ────────────────────────────────
def test_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint, num_labels=NUM_METRICS,
        torch_dtype=torch.bfloat16
    ).eval().cuda()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    rows = load_jsonl(args.train_path)
    train_rows, dev_rows = train_test_split(rows, test_size=0.4, random_state=RANDOM_SEED)
    dev_rows, test_rows = train_test_split(dev_rows, test_size=0.5, random_state=RANDOM_SEED)

    prompt_template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        if "llama" in args.model_checkpoint.lower() else "{system_prompt}"
    )

    data = [
        {
            "prompt": prompt_template.format(
                system_prompt=r["prompt"].replace(" /// ", " ")
            ),
            "scores": [r[m] for m in METRIC_COLS],   # ← 直接用原始分值
        }
        for r in test_rows
    ]
    ds = Dataset.from_list(data)

    def tok_fn(ex):
        out = tokenizer(ex["prompt"], truncation=True, max_length=args.max_length)
        out["scores"] = ex["scores"]
        return out

    ds = ds.map(tok_fn, remove_columns=ds.column_names)

    def collate(batch):
        return {
            "input_ids": torch.tensor([b["input_ids"] for b in batch]),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
            "scores": torch.tensor([b["scores"] for b in batch]),
        }

    loader = DataLoader(ds, batch_size=args.eval_batch_size, collate_fn=collate)

    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            logits = model(
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
            ).logits.cpu()
            preds.append(logits)
            trues.append(batch["scores"])

    preds = torch.cat(preds).to(torch.float32).numpy()
    trues = torch.cat(trues).to(torch.float32).numpy()

    print("\nSpearman correlation per metric")
    for i, m in enumerate(METRIC_COLS):
        ro, p = spearmanr(rankdata(-preds[:, i]), rankdata(-trues[:, i]))
        print(f"{m:15s}: ρ = {ro:.4f}  (p = {p:.3g})")
    

    # calculate spearman correlation of overall scores
    weights = {
            "acc_mean": 0.5,
            "acc_var": -0.25,
            "output_tokens_var": -0.125,
            "consistency": 0.125
        }
    overall_pred = 0
    overall_true = 0
    overall_sigmoid_pred = 0
    for i, metric in enumerate(METRIC_COLS):
        overall_pred += weights[metric] * preds[:, i]
        overall_sigmoid_pred += weights[metric] * 1 / (1 + np.exp(-preds[:, i]))
        overall_true += weights[metric] * trues[:, i]

    ro, p = spearmanr(rankdata(-overall_pred), rankdata(-overall_true))
    print(f"Overall score: ρ = {ro:.4f}  (p = {p:.3g})")
    ro, p = spearmanr(rankdata(-overall_sigmoid_pred), rankdata(-overall_true))
    print(f"Overall sigmoid score: ρ = {ro:.4f}  (p = {p:.3g})")


# ─────────────────────────────── main ───────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Multi-metric reward model")
    p.add_argument("--mode", choices=["train", "test"], required=True)
    p.add_argument("--model_checkpoint", type=str, required=True)
    p.add_argument("--train_path", type=str, help="JSONL with training rows")
    p.add_argument("--test_path", type=str, help="JSONL with test rows")
    p.add_argument("--output_dir", type=str, default="./out")
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "train":
        if args.train_path is None:
            raise ValueError("`--train_path` is required in train mode")
        train_model(args)
    else:
        # if args.test_path is None:
        #     raise ValueError("`--test_path` is required in test mode")
        test_model(args)

if __name__ == "__main__":
    main()
