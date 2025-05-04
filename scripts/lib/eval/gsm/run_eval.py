import argparse
import os
import re
import json
import random
from eval.utils import (
    generate_completions,
    load_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
    dynamic_import_function,
    ensure_dir
)
from eval.math_util import my_answer_extraction
from eval.math_equivalence import is_equiv


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output


def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    with open(os.path.join(args.data_dir, "test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["question"],
                "answer": str(example["answer"])
            })

    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]

    ensure_dir(args.save_dir)

    prompt_prefix = "Solve the following math problem. Put your final answer within \\boxed{}.\n\n"

    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for example in test_data:
        prompt = prompt_prefix + "Question: " + example["question"].strip()
        prompts.append(prompt)

    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    if args.model_name_or_path:
        assert 0, "Deprecated"
        print("Loading model and tokenizer...")
        model, tokenizer = load_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    elif args.base_model_name_or_path:
        model, tokenizer = load_dexperts_model_and_tokenizer(
            args.base_model_name_or_path,
            args.expert_model_name_or_path,
            args.anti_expert_model_name_or_path,
            chat_response_prefix="Answer:",
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
            log_file=os.path.join(args.save_dir, "logits.log"),
            alpha=args.alpha,
            alpha_strategy=args.alpha_strategy
        )
    print("Finish loading model and tokenizer!")

    if "Qwen" in args.base_model_name_or_path:
        stop_id_sequences = [[151643], [151645]]
    else:
        stop_id_sequences = None

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
        do_sample=args.do_sample,
        stop_id_sequences=stop_id_sequences,
        stop_repetitive_generation=args.alpha_strategy is not None,
    )


    predictions = [my_answer_extraction(output) for output in outputs]
    targets = [example["answer"] for example in test_data]
    assert len(predictions) == len(targets)

    em_score = sum([int(is_equiv(pred, answer)) for pred, answer in zip(predictions, targets)]) / len(targets)

    print(f"Exact match : {em_score}")

    predictions = [{
        "question": example["question"],
        "answer": example["answer"],
        "model_output": output,
        "prediction": pred
    } for example, output, pred in zip(test_data, outputs, predictions)]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump({
            "exact_match": em_score
        }, fout, indent=4)


if __name__ == "__main__":
    def str_to_bool(value):
        if isinstance(value, bool):  # If already a bool, return it directly
            return value
        if value.lower() in ("true", "1", "yes", "t"):
            return True
        elif value.lower() in ("false", "0", "no", "f"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected (True/False).")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/gsm"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="initial alpha"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="DEPRECATED if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="max generation tokens"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--anti_expert_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--alpha_strategy",
        type=str,
        default=None
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--do_sample",
        type=str_to_bool,
        required=True,
        help="sample (True) or greedy (False)"
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    args = parser.parse_args()

    main(args)
