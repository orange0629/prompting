from lib.dataloader import init_benchmark
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Script to run predictions with a specified GPU and data file.")
    parser.add_argument("-benchmark", help="Benchmark to evaluate", type=str, default="truthfulqa")
    parser.add_argument("-raw_pred_file", help="Raw Prediction File", type=str, required=True)
    parser.add_argument("-prompt_score_file", help="Prompt Score File", type=str, required=True)
    return parser.parse_args()

args = parse_args()

benchmark_obj = init_benchmark(name=args.benchmark)
benchmark_obj.eval_saved_file(args.raw_pred_file, args.prompt_score_file, metric_list=["bleu"])