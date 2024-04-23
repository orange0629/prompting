from lib.dataloader import init_benchmark

benchmark="truthfulqa"
raw_pred_file = "./results/benchmark/Qwen1.5-0.5B-Chat_truthfulqa_results.csv"
prompt_score_file = "./data/system_prompts/Prompt-Scores_Good-Property.csv"


benchmark_obj = init_benchmark(name=benchmark)
benchmark_obj.eval_saved_file(raw_pred_file, prompt_score_file, metric_list=["bleu"])