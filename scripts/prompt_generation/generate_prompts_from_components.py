import json
import random

# 设置随机种子，确保每次运行结果一致
random.seed(42)

# 参数
input_file = "../../data/system_prompts/generated_prompt_components_20250315_zh.jsonl"  # 输入 JSONL 文件
output_file = input_file.replace("prompt_components", "prompt")
num_combinations = 10000  # 生成的组合数量
max_length = 30  # 组合的最大长度

# 读取 JSONL 文件
components = []
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        data = json.loads(line.strip())  # 解析 JSONL 行
        components.append((data["prompt"], data["category"]))

# 生成索引列表（确保相同数据输入时索引顺序一致）
index_list = list(range(len(components)))

# 生成随机组合
seen_combinations = set()
generated_data = []

while len(generated_data) < num_combinations:
    # 根据给定的概率分布选择长度
    length = min(max_length, random.choices(
        range(1, max_length + 1),
        weights=[1 / (i ** 0.8) for i in range(1, max_length + 1)]
    )[0])

    # 随机选择组件索引
    selected_indices = random.sample(index_list, min(length, len(components)))

    # 生成组合
    combined_prompt = " /// ".join([components[i][0] for i in selected_indices])
    combined_category = [components[i][1] for i in selected_indices]

    # 组合成不可变 tuple 进行去重
    combination_tuple = (tuple(combined_category), combined_prompt)

    if combination_tuple not in seen_combinations:
        seen_combinations.add(combination_tuple)
        generated_data.append({"prompt": combined_prompt, "category": combined_category})

# 保存 JSONL 文件
with open(output_file, "w", encoding="utf-8") as outfile:
    for item in generated_data:
        outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"生成完成，已保存到 {output_file}")