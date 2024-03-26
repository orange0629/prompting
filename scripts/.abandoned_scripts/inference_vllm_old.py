from vllm import LLM
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score

model_dir = "/shared/4/models/llama2/pytorch-versions/llama-2-7b-chat/"
data_dir = "../data/mmlu/mmlu_mingqian.csv"
system_prompts_dir = "../data/system_prompts/Generated-Prompt_Good-Property_gpt-3.5-turbo.csv"
cache_dir= "/shared/4/models/"

data_df = pd.read_csv(data_dir)
system_prompts_df = pd.read_csv(system_prompts_dir)
system_prompts = system_prompts_df["Prompt"]
llm = LLM(model=model_dir)  # Create an LLM.

# Here's Mingqian's prompt
#template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{{You will be presented with a role-playing context followed by a multiple-choice question. {role_context} Select only the option number that corresponds to the correct answer for the following question.}}\n\n### Input:\n{{{{{question}}} Provide the number of the correct option without explaining your reasoning.}} \n\n### Response:'''
#flan_template = '''{role_context} {question} Please select the correct answer number:'''
#role_context = "You are a helpful assistant."

user_prompt = "The following is a multiple choice question (with answers). Reply with only the option letter.\n" + "{question_prompt}"
llama_template = '''[INST] <<SYS>>
{system_prompt}
<</SYS>>
{user_prompt}[/INST]Answer:'''

letter2num = {"A": 1, "B": 2, "C": 3, "D": 4, "Z": 5}
acc_list = []
error_list = []

for system_prompt in tqdm(system_prompts):

    answer_prompts = []
    for idx, item in data_df.iterrows():
        question_text = item['question']
        option1 = item["option1"]
        option2 = item["option2"]
        option3 = item["option3"]
        option4 = item["option4"]

        question_prompt = f"{question_text.strip()}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}\n"
        #choices_text = f'Options: 1. {option1}, 2. {option2}, 3. {option3}, 4. {option4}.'
        full_prompt = llama_template.format(system_prompt=system_prompt, user_prompt=user_prompt.format(question_prompt=question_prompt))
        answer_prompts.append(full_prompt)
    
    outputs = llm.generate(answer_prompts)  # Generate texts from the prompts.

    model_answer_list = []
    errors = 0
    for output in outputs:
        text = output.outputs[0].text.replace("\n", "").strip()
        model_choice_tmp = "Z"
        for c in text:
            if c in ["A", "B", "C", "D"]:
                model_choice_tmp = c
                break
        model_answer_list.append(letter2num[model_choice_tmp])
        if model_choice_tmp == "Z":
            errors += 1
    
    acc_list.append(accuracy_score(data_df["true_option"], model_answer_list))
    error_list.append(errors)

system_prompts_df["MMLU_acc"] = acc_list
system_prompts_df["MMLU_error"] = error_list
system_prompts_df.to_csv("good_property_mmlu.csv", index=False)