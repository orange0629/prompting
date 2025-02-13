import requests
import os
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
import evaluators
import concurrent.futures
from tqdm import tqdm
import time
import json
import argparse
import scorers
import tasks
import predictors
import optimizers


def get_task_class(task_name):
    if task_name == 'ethos':
        return tasks.EthosBinaryTask
    elif task_name == 'jailbreak':
        return tasks.JailbreakBinaryTask
    elif task_name == 'liar':
        return tasks.DefaultHFBinaryTask
    elif task_name == 'ar_sarcasm':
        return tasks.DefaultHFBinaryTask
    else:
        return tasks.CustomTask(task_name)


def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')



def get_scorer(scorer):
    if scorer == '01':
        return scorers.Cached01Scorer
    elif scorer == 'll':
        return scorers.CachedLogLikelihoodScorer
    elif scorer == 'custom':
        return scorers.CustomScorer
    else:
        raise Exception(f'Unsupported scorer: {scorer}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='bbh_boolean_expressions')
    parser.add_argument('--model_dir', default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--prompts', default='prompts/bbh_boolean_expressions.md')
    parser.add_argument('--system_prompt', default="no_sys")
    # parser.add_argument('--config', default='default.json')
    parser.add_argument('--out', default=f'protegi_{str(int(time.time()))}.txt')
    parser.add_argument('--max_threads', default=32, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=6, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_test_exs', default=400, type=int)

    parser.add_argument('--minibatch_size', default=64, type=int)
    parser.add_argument('--n_gradients', default=4, type=int)
    parser.add_argument('--errors_per_gradient', default=4, type=int)
    parser.add_argument('--gradients_per_error', default=1, type=int)
    parser.add_argument('--steps_per_gradient', default=1, type=int)
    parser.add_argument('--mc_samples_per_step', default=2, type=int)
    parser.add_argument('--max_expansion_factor', default=8, type=int)

    parser.add_argument('--engine', default="chatgpt", type=str)

    parser.add_argument('--evaluator', default="bf", type=str)
    parser.add_argument('--scorer', default="custom", type=str)
    parser.add_argument('--eval_rounds', default=8, type=int)
    parser.add_argument('--eval_prompts_per_round', default=8, type=int)
    # calculated by s-sr and sr
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)
    parser.add_argument('--reject_on_errors', action='store_true') 
    parser.add_argument('--multi_gpu', default=1, type=int) 
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    model_name = args.model_dir.split("/")[-1]
    saving_dir = os.path.join(project_root_dir, f"./data/task_prompts/{args.task}")
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    args.out = os.path.join(saving_dir, f'protegi_{model_name}_{args.system_prompt}_log_{str(int(time.time()))}.log')
    args.prompts = os.path.join(saving_dir, "./base.md")

    with open(os.path.join(project_root_dir, f'./data/system_prompts/{args.system_prompt}.md'), 'r') as file:
        system_prompt = file.read()

    config = vars(args)

    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']
    
    task = get_task_class(args.task)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)
    gpt4 = predictors.BinaryPredictor(config)
    llama_vllm = predictors.VLLMPredictor(args.model_dir, args.multi_gpu, system_prompt=system_prompt)

    optimizer = optimizers.ProTeGi(
        config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    test_exs = task.get_test_examples()

    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(config) + '\n')

    candidates = [open(fp.strip()).read() for fp in args.prompts.split(',')]

    for round in tqdm(range(config['rounds'] + 1)):
        print("STARTING ROUND ", round)
        start = time.time()

        # expand candidates
        if round > 0:
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs, llama_vllm)

        # score candidates
        scores = optimizer.score_candidates(candidates, task, llama_vllm, train_exs)
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

        # select candidates
        candidates = candidates[:config['beam_size']]
        scores = scores[:config['beam_size']]

        # record candidates, estimated scores, and true scores
        with open(args.out, 'a') as outf:
            outf.write(f"======== ROUND {round}\n")
            outf.write(f'{time.time() - start}\n')
            outf.write(f'{candidates}\n')
            outf.write(f'{scores}\n')
        metrics = []
        for candidate, score in zip(candidates, scores):
            f1, texts, labels, preds, _ = task.evaluate(llama_vllm, candidate, test_exs, n=args.n_test_exs)
            metrics.append(f1)
        with open(args.out, 'a') as outf:  
            outf.write(f'{metrics}\n')

    print("DONE!")

    if args.system_prompt == "no_sys":
        with open(os.path.join(saving_dir, f"protegi_{model_name}.md"), 'w', encoding='utf-8') as file:
            file.write(candidates[0])
    else:
        with open(os.path.join(saving_dir, f"protegi_{model_name}_{args.system_prompt}.md"), 'w', encoding='utf-8') as file:
            file.write(candidates[0])