import openai
import json
import random
import time
from typing import List, Tuple
import pandas as pd

RUN_NAME = "20250127"

client = openai.OpenAI(
    api_key="",
)
# client = openai.AzureOpenAI(
#   azure_endpoint = "https://api.umgpt.umich.edu/azure-openai-api",
#   azure_deployment = "gpt-4o",
#   api_key="4a86861856264afeaf59fc76b8d12376",
#   api_version="2024-10-21"
# )

# API调用函数（带重试机制）
def call_chatgpt(prompt: str, max_retries=3) -> str:
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt},
                    ]
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error: {e}, retrying...")
            time.sleep(2)
    return ""

# print(call_chatgpt("Hi there."))
# quit()

def log_component_history(component: str, action: str, parents: List[str], parameters: dict):
    """记录组件生成历史到 JSONL 文件"""
    entry = {
        "component": component,
        "action": action,
        "parents": parents,
        "parameters": parameters
    }
    with open(f'component_history_{RUN_NAME}.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# 组件生成函数
def generate_components(prompt: str, max_retries=3) -> List[str]:
    retries = 0
    while retries < max_retries:
        try:
            response = call_chatgpt(prompt)
            return eval(response[response.find('['):response.rfind(']') + 1])
        except Exception as e:
            retries += 1
            print(f"Error expanding prompt (attempt {retries}/{max_retries}): {type(e).__name__}: {str(e)}")
            time.sleep(3)
    print("Failed to expand prompt after maximum retries.")
    return []


# 操作概率配置（可调整）
ACTION_PROB = {
    'add_useful': 0.1,
    'add_useless': 0.05,
    'refine_subset': 0.1,
    'rephrase_subset': 0.1,
    'no_action': 0.65
}

# 新增动态处理函数
def dynamic_process(selected: List[str], components: List[str]) -> Tuple[List[str], List[str]]:
    """
    对当前选择的组件进行动态处理，返回(modified_selected, new_components)
    """
    action = random.choices(
        list(ACTION_PROB.keys()),
        weights=list(ACTION_PROB.values()),
        k=1
    )[0]
    
    new_components = []
    
    if action == 'add_useful':
        # 基于当前选择生成有用组件
        prompt = f"""You are an expert in optimizing system prompts for LLMs to enhance their general performance. \
Given the following list of system prompt components: {json.dumps(selected)}, generate 1-2 additional components \
that can further improve the LLM's capabilities. Return the result strictly as a Python list of strings. \
No additional explanations or formatting, only return the list."""
        new = generate_components(prompt)
        for c in new:
            log_component_history(
                component=c,
                action=action,
                parents=selected.copy(),
                parameters={"prompt": prompt}
            )
        selected += new
        new_components.extend(new)
        
    elif action == 'add_useless':
        # 添加无用组件
        prompt = f"""Given the following list of system prompt components: {json.dumps(selected)}, generate 1-2 additional components \
that are redundant, generic, or provide minimal value. Examples: ["Answer in English.", "Be polite."]. Return the result strictly \
as a Python list of strings. No additional explanations or formatting, only return the list."""
        new = generate_components(prompt)
        for c in new:
            log_component_history(
                component=c,
                action=action,
                parents=selected.copy(),
                parameters={"prompt": prompt}
            )
        selected += new
        new_components.extend(new)
        
    elif action == 'refine_subset' and len(selected)>=2:
        # 精炼子集为单个组件
        subset = random.sample(selected, min(random.randint(2, 5), len(selected)))
        prompt = f"""Given the following list of sentences: {json.dumps(selected)}, combine these into one concise \
sentence. No additional explanations or formatting, only return a sentence."""
        refined = call_chatgpt(prompt)
        if refined:
            log_component_history(
                component=refined,
                action=action,
                parents=subset,
                parameters={
                    "prompt": prompt,
                    "subset_size": len(subset),
                    "subset_components": subset
                }
            )
            selected = [c for c in selected if c not in subset] + [refined]
            new_components.append(refined)
            
    elif action == 'rephrase_subset' and len(selected)>=1:
        # 重写子集组件
        subset = random.sample(selected, min(random.randint(1, 5), len(selected)))
        rephrased = []
        for c in subset:
            prompt = f"""Rephrase this sentence keeping the same meaning: {c}. \
No additional explanations or formatting, only return a sentence."""
            new_c = call_chatgpt(prompt)
            if new_c:
                log_component_history(
                    component=new_c,
                    action=action,
                    parents=c,
                    parameters={
                        "original": c,
                        "prompt": prompt
                    }
                )
                rephrased.append(new_c)
        if rephrased:
            selected = [c for c in selected if c not in subset] + rephrased
            # new_components.extend(rephrased)
    
    # 去重并更新全局组件列表
    unique_new = [c for c in new_components if c not in components]
    components.extend(unique_new)
    
    return list(set(selected)), unique_new

# 修改后的组合生成函数
def generate_combinations_dynamic(components: List[str], 
                                 target_count=100000, 
                                 max_length=30) -> List[str]:
    combinations = []
    component_set = set(components)  # 用于快速查重
    
    while len(combinations) < target_count:
        # 动态调整组件列表
        components = list(component_set)
        
        # 随机选择长度
        length = min(max_length, random.choices(
            range(1, max_length+1), 
            weights=[1/(i**0.8) for i in range(1, max_length+1)]
        )[0])
        
        # 初始选择
        selected = random.sample(components, min(length, len(components)))
        
        # 动态处理
        processed_selected, new_components = dynamic_process(selected, components)
        
        # 更新全局组件集合
        component_set.update(new_components)
        
        # 打乱并生成最终prompt
        random.shuffle(processed_selected)
        final_prompt = " /// ".join(processed_selected)
        
        combinations.append(final_prompt)
        
        # 进度报告
        if len(combinations) % 100 == 0:
            print(f"Generated: {len(combinations)} | Unique components: {len(component_set)}")
            if len(new_components) > 0:
                print(f"New components added: {new_components[:3]}...")
            
            with open(f'dynamic_components_{RUN_NAME}.json', 'w') as f:
                json.dump(list(component_set), f, indent=2)

            with open(f'dynamic_prompts_{RUN_NAME}.json', 'w') as f:
                json.dump(combinations, f)
            
    
    return combinations, list(component_set)

# 初始化组件（示例）
components = list(pd.read_csv("./data/system_prompts/Prompt_Component_Corpus.csv")["Prompt"])

# 运行动态生成流程
final_dataset, expanded_components = generate_combinations_dynamic(components)

# 保存结果
with open(f'dynamic_components_{RUN_NAME}.json', 'w') as f:
    json.dump(expanded_components, f, indent=2)

with open(f'dynamic_prompts.json_{RUN_NAME}', 'w') as f:
    json.dump(final_dataset, f)