import json
from translate import translate_text
from tqdm import tqdm


# 读取 JSONL 文件并翻译 prompt 字段
input_file = "../../data/system_prompts/generated_prompt_components_20250315_sup.jsonl"
output_file = input_file.replace(".jsonl", "_zh.jsonl")

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile):
        data = json.loads(line.strip())  # 解析 JSONL 行
        data["prompt"] = translate_text(data["prompt"])  # 翻译 prompt
        print(data["prompt"])
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")  # 写回 JSONL 文件

print(f"翻译完成，结果已保存至 {output_file}")