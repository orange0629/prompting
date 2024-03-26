from vllm import LLM
import pandas as pd
from tqdm import tqdm
from lib.dataloader import init_benchmark

model_dir = "/shared/4/models/llama2/pytorch-versions/llama-2-7b-chat/"
benchmark = "arc"
system_prompts_dir = "./data/system_prompts/Prompt-Scores_Good-Property.csv"
cache_dir= "/shared/4/models/"

system_prompts_df = pd.read_csv(system_prompts_dir)
system_prompts = system_prompts_df["Prompt"]
llm = LLM(model=model_dir)  # Create an LLM.

# Here's Mingqian's prompt
#template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{{You will be presented with a role-playing context followed by a multiple-choice question. {role_context} Select only the option number that corresponds to the correct answer for the following question.}}\n\n### Input:\n{{{{{question}}} Provide the number of the correct option without explaining your reasoning.}} \n\n### Response:'''
#flan_template = '''{role_context} {question} Please select the correct answer number:'''
#role_context = "You are a helpful assistant."

user_prompt = "The following is a multiple choice question (with answers). Reply with only the option letter.\n{question_prompt}"
llama_template = '''[INST] <<SYS>>
{system_prompt}
<</SYS>>
{user_prompt}[/INST]Answer:'''

acc_list = []
error_list = []
benchmark_obj = init_benchmark(name=benchmark)
q_list = benchmark_obj.load_question_list()

for system_prompt in tqdm(system_prompts):
    answer_prompts = []
    for q in q_list:
        full_prompt = llama_template.format(system_prompt=system_prompt, user_prompt=user_prompt.format(question_prompt=q))
        answer_prompts.append(full_prompt)
    outputs = llm.generate(answer_prompts)  # Generate texts from the prompts.
    score, error_num = benchmark_obj.eval_question_list(outputs, vllm=True)
    acc_list.append(score)
    error_list.append(error_num)

system_prompts_df[f"{benchmark.upper()}_acc"] = acc_list
system_prompts_df[f"{benchmark.upper()}_error"] = error_list
system_prompts_df.to_csv(system_prompts_dir, index=False)