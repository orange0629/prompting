import argparse
import numpy as np
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from datasets import load_dataset, Dataset
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from itertools import combinations
from sklearn.model_selection import train_test_split
import random
#from trl import RewardTrainer, RewardConfig
from dataclasses import dataclass
from trl import RewardConfig
from trl.trainer.reward_trainer import *

COL_NAME = "Meta-Llama-3.1-8B-Instruct/avg_score"

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    return data


def generate_comparisons(data, prompt_template):
    comparisons = []
    for item1, item2 in combinations(data, 2):
        if item1[COL_NAME] > item2[COL_NAME]:
            chosen, rejected = item1, item2
        else:
            chosen, rejected = item2, item1

        # comparisons.append({
        #     "example_chosen": tokenizer.apply_chat_template([{"role": "system", "content": chosen["prompt"].replace(" /// ", " ")}], tokenize=False, add_generation_prompt=False),
        #     "example_rejected": tokenizer.apply_chat_template([{"role": "system", "content": rejected["prompt"].replace(" /// ", " ")}], tokenize=False, add_generation_prompt=False),
        # })
        comparisons.append({
            "example_chosen": prompt_template.format(system_prompt=chosen["prompt"].replace(" /// ", " ")),
            "example_rejected": prompt_template.format(system_prompt=rejected["prompt"].replace(" /// ", " ")),
            "score_chosen": chosen[COL_NAME],
            "score_rejected": rejected[COL_NAME],
        })
    random.shuffle(comparisons)
    return Dataset.from_list(comparisons)

@dataclass
class RewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features_chosen = []
        features_rejected = []
        scores_chosen = []
        scores_rejected = []
        margin = []
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if "score_chosen" in feature and "score_rejected" in feature:
                scores_chosen.append(feature["score_chosen"])
                scores_rejected.append(feature["score_rejected"])
            if has_margin:
                margin.append(feature["margin"])
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        if len(scores_chosen) > 0 and len(scores_rejected) > 0:
            batch["score_chosen"] = torch.tensor(scores_chosen, dtype=torch.float)
            batch["score_rejected"] = torch.tensor(scores_rejected, dtype=torch.float)
        return batch

class PromptRewardTrainer(RewardTrainer):
    def visualize_samples(self, num_print_samples: int):
        """
        Visualize the reward model logits prediction

        Args:
            num_print_samples (`int`, defaults to `4`):
                The number of samples to print. Set to `-1` to print all samples.
        """
        num_print_samples = 100 # Manually set by Lechen
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        for _, inputs in enumerate(eval_dataloader):
            _, logits, _ = self.prediction_step(self.model, inputs, prediction_loss_only=False)
            chosen_text = decode_and_strip_padding(inputs["input_ids_chosen"], self.processing_class)
            rejected_text = decode_and_strip_padding(inputs["input_ids_rejected"], self.processing_class)
            table["chosen_text"].extend(gather_object(chosen_text))
            table["rejected_text"].extend(gather_object(rejected_text))
            table["logits"].extend(
                gather_object([[round(inner_item, 4) for inner_item in item] for item in logits.tolist()])
            )
            table["score_chosen"].extend(gather_object(inputs["score_chosen"]).cpu())
            table["score_rejected"].extend(gather_object(inputs["score_rejected"]).cpu())
            table["margin"].extend(gather_object(inputs["margin"]).cpu())
            
            if num_print_samples >= 0 and len(table["chosen_text"]) >= num_print_samples:
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df[:min(num_print_samples, 3)])
            if "wandb" in self.args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in self.args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )

def train_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint, num_labels=1, torch_dtype=torch.bfloat16, use_flash_attention_2=True, cache_dir="/shared/4/models/"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    data = load_jsonl(args.train_path)
    train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)

    if "llama" in args.model_checkpoint.lower():
        prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    else:
        prompt_template = "{system_prompt}"
    train_dataset = generate_comparisons(train_data, prompt_template).select(range(100000))
    test_dataset = generate_comparisons(test_data, prompt_template).select(range(1000))

    def tokenize_function(examples):
        tokens_chosen = tokenizer(examples["example_chosen"], padding='max_length', truncation=True, max_length=args.max_length)
        tokens_rejected = tokenizer(examples["example_rejected"], padding='max_length', truncation=True, max_length=args.max_length)
        
        return {
            "input_ids_chosen": tokens_chosen["input_ids"],
            "attention_mask_chosen": tokens_chosen["attention_mask"],
            "input_ids_rejected": tokens_rejected["input_ids"],
            "attention_mask_rejected": tokens_rejected["attention_mask"],
            "score_chosen": examples["score_chosen"],
            "score_rejected": examples["score_rejected"],
            "margin": list((np.array(examples["score_chosen"]) - np.array(examples["score_rejected"]))*100),
        }

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    training_args = RewardConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=10,
        num_train_epochs=1,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        bf16=True,
        logging_strategy="steps",
        logging_steps=1, # warmup_ratio=0.03,
        report_to='wandb',
        max_length=args.max_length,
    )

    trainer = PromptRewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=RewardDataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)


def test_model(args):
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint, num_labels=1, torch_dtype=torch.bfloat16, use_flash_attention_2=True
    )
    model.to("cuda")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    data = load_jsonl(args.test_path)
    train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)

    def tokenize_function(examples):
        return tokenizer(examples["prompt"], padding='max_length', truncation=True, max_length=args.max_length)
    
    test_dataset_df = pd.DataFrame(test_data)[["prompt", COL_NAME]]
    test_dataset_df["prompt"] = test_dataset_df["prompt"].apply(lambda x: x.replace(" /// ", " "))
    test_dataset = Dataset.from_pandas(test_dataset_df)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["prompt"])
    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=args.eval_batch_size, collate_fn=lambda batch: {key: torch.stack([torch.tensor(item[key]) for item in batch]) for key in batch[0]})

    pred_score_lst = []
    true_score_lst = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            outputs = model(input_ids=batch["input_ids"].to("cuda"), attention_mask=batch["attention_mask"].to("cuda"))
            pred_score_lst.extend(outputs["logits"].reshape(-1).cpu().tolist())
            true_score_lst.extend(batch[COL_NAME].cpu().tolist())
            # predicted_label = label_encoder.inverse_transform([predicted_label_


    from scipy.stats import rankdata, spearmanr
    pred_rank_lst = rankdata([-x for x in pred_score_lst])
    true_rank_lst = rankdata([-x for x in true_score_lst])
    spearman_corr, p_value = spearmanr(pred_rank_lst, true_rank_lst)
    print(f"Spearman's Rank Correlation: {spearman_corr}")
    print(f"P-value: {p_value}")


def main():
    parser = argparse.ArgumentParser(description="Train and Test LLaVA model on Mastodon Dataset")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Mode: train or test")
    # parser.add_argument('--base_model', type=str, required=True, help="Path to base model")
    parser.add_argument('--model_checkpoint', type=str, help="Path to the model checkpoint")
    parser.add_argument('--train_path', type=str, help="Path to the training dataset (JSONL)")
    parser.add_argument('--dev_path', type=str, help="Path to the dev dataset (JSONL)")
    parser.add_argument('--test_path', type=str, help="Path to the test dataset (JSONL)")
    parser.add_argument('--output_dir', type=str, help="Path to save the trained model")
    # parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument('--train_batch_size', type=int, default=1, help="Training batch size")
    parser.add_argument('--eval_batch_size', type=int, default=1, help="Evaluation batch size")
    parser.add_argument('--max_length', type=int, default=128)

    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)

if __name__ == "__main__":
    main()