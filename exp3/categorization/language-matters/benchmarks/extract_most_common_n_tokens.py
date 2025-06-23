import json
from colorama import Fore, Style
from collections import defaultdict
from transformers import AutoTokenizer
from argparse import ArgumentParser, Namespace

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=False, default=None)
    return parser.parse_args()

IGNORE_TOKENS = 0  # whether to ignore the first <IGNORE_TOKENS> tokens

def main(args: Namespace):
    double_decode = False
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        double_decode = True

    first_n_token_string_count = {i: defaultdict(int) for i in range(args.n)}
    first_n_token_combination_count = {i: defaultdict(int) for i in range(args.n)}
    with open(args.input_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            if "logprobs" in data:
                token2logprob_seq = data["logprobs"]
                if len(token2logprob_seq) == 0:
                    continue
            else:
                token2logprob_seq = [ {r: 0} for r in data["output"][:1000]]
            string = ""
            tokens = []
            for i in range(IGNORE_TOKENS, args.n):
                token2logprob = token2logprob_seq[i]
                if type(token2logprob) == dict:
                    token, _ = list(token2logprob.items())[0]
                elif type(token2logprob) == list:
                    token = token2logprob[0]["token"]
                else:
                    raise TypeError(f"Unexpected type: {type(token2logprob)}")
                if double_decode:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    tokens.append(token_id)
                string += token
                first_n_token_string_count[i][string] += 1
                first_n_token_combination_count[i]['-'.join([ str(r) for r in tokens])] += 1
    for i in range(IGNORE_TOKENS, args.n):
        print(Fore.GREEN + f"Top {i + 1 - IGNORE_TOKENS} tokens:" + Style.RESET_ALL)
        for string, count in sorted(first_n_token_string_count[i].items(), key=lambda x: x[1], reverse=True):
            print(f"{string}: {Fore.RED}{count}{Style.RESET_ALL}")
        if double_decode:
            for token_string, count in sorted(first_n_token_combination_count[i].items(), key=lambda x: x[1], reverse=True):
                token_str_list = token_string.split('-')
                tokens = [ int(s) for s in token_str_list]
                string = tokenizer.decode(tokens)
                print(f"{string}: {Fore.RED}{count}{Style.RESET_ALL}")
        print()


if __name__ == "__main__":
    args = setup_args()
    main(args)
