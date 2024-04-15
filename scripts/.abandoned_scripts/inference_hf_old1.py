from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

BATCH_SIZE = 8

model_dir = "/shared/4/models/llama2/pytorch-versions/llama-2-7b-chat/"
data_dir = "../data/mmlu/mmlu_mingqian.csv"
cache_dir= "/shared/4/models/"

data_df = pd.read_csv(data_dir)

template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{{You will be presented with a role-playing context followed by a multiple-choice question. {role_context} Select only the option number that corresponds to the correct answer for the following question.}}\n\n### Input:\n{{{{{question}}} Provide the number of the correct option without explaining your reasoning.}} \n\n### Response:'''
flan_template = '''{role_context} {question} Please select the correct answer number:'''
role_context = "You are a helpful assistant."

answer_prompts = []
for idx, item in data_df.iterrows():
    question_text = item['question']
    option1 = item["option1"]
    option2 = item["option2"]
    option3 = item["option3"]
    option4 = item["option4"]

    choices_text = f'Options: 1. {option1}, 2. {option2}, 3. {option3}, 4. {option4}.'
    question_text = f"{question_text} {choices_text}"
    full_prompt = template.format(role_context=role_context, question=question_text)
    answer_prompts.append(full_prompt)

tokenizer = LlamaTokenizer.from_pretrained(model_dir, 
                                           cache_dir=cache_dir,
                                           padding_side='left',
                                           )

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = LlamaForCausalLM.from_pretrained(model_dir, 
                                         cache_dir=cache_dir,
                                         device_map="auto",
                                         #quantization_config=quantization_config,
                                         #load_in_8bit=True
                                        )


BATCH_SIZE = 16
answer_list = []

for idx in tqdm(range(0, len(answer_prompts), BATCH_SIZE)):
    ques_batch = answer_prompts[idx:(idx+BATCH_SIZE)]
    ques_batch_tokenized = tokenizer(ques_batch, return_tensors='pt', truncation=True, max_length=512, padding=True)
    answ_ids = model.generate(**ques_batch_tokenized.to('cuda'), max_new_tokens=30, pad_token_id=tokenizer.pad_token_id)
    answer_list.extend(tokenizer.batch_decode(answ_ids, skip_special_tokens=True))

print(answer_list[:10])
