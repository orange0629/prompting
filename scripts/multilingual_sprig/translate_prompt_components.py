import json
from translate import translate_text
from tqdm import tqdm

target_lang = "es"


input_file = "../../data/system_prompts/generated_prompt_components_20250315.jsonl"
output_file = input_file.replace(".jsonl", f"_{target_lang}.jsonl")

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile):
        data = json.loads(line.strip())  # 解析 JSONL 行
        data["prompt"] = translate_text(data["prompt"], target=target_lang)  # 翻译 prompt
        print(data["prompt"])
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")  # 写回 JSONL 文件

print(f"Translation done. Saved at {output_file}")