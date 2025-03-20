import requests
import os

API_KEY = ""

import pandas as pd

languages = {
    "en": "English",
    "zh": "Chinese",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "ar": "Arabic",
    "bn": "Bengali",
    "pt": "Portuguese",
    "ru": "Russian",
    "ur": "Urdu",
}

answer_identifiers = {
    "en": "Answer:",
    "zh": "答案：",
    "hi": "उत्तर:",
    "es": "Respuesta:",
    "fr": "Réponse:",
    "ar": "الإجابة:",
    "bn": "উত্তর:",
    "pt": "Resposta:",
    "ru": "Ответ:",
    "ur": "جواب:",
}

def translate_text(text, source="en", target="zh-CN"):
    if target == "zh":
        target = "zh-CN"
    url = f"https://translation.googleapis.com/language/translate/v2?key={API_KEY}"
    data = {
        "q": text,
        "source": source,
        "target": target,
        "format": "text",
    }
    
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()["data"]["translations"][0]["translatedText"]
    else:
        print(f"Translation failed: {response.text}")
        return text  

# input_file = "gsm8k.csv"
# output_file = "output.csv"


# df = pd.read_csv(input_file)

# df.loc[:19, "translated_question"] = df.iloc[:20, 0].astype(str).apply(translate_text)
# df.loc[:19, "translated_answer"] = df.iloc[:20, 1].astype(str).apply(translate_text)


# df.to_csv(output_file, index=False)

# print(f"Translation completed.")

def translate_markdown_files(folder_path, target_languages):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith("old.md"):
                file_path = os.path.join(root, file_name)
                for target_language in target_languages:
                    translated_file_path = os.path.join(
                        root, f"{os.path.splitext(file_name)[0]}_{target_language}.md"
                    )
                    
                    if os.path.exists(translated_file_path):
                        print(f"Translated file already exists, skipping: {translated_file_path}")
                        continue
                    
                    print(f"Processing file: {file_path} for language: {target_language}")
                    
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    content = content.replace(answer_identifiers["en"], answer_identifiers[target_language])
                    translated_content = translate_text(content, target_language) if target_language != "en" else content

                    if answer_identifiers[target_language] not in content:
                        print(f"Lost answer identifier: {file_name} for language: {target_language}")
                        continue

                    if translated_content is None:
                        print(f"Failed to translate file: {file_name} for language: {target_language}")
                        continue
                    
                    # Save translated content
                    with open(translated_file_path, "w", encoding="utf-8") as f:
                        f.write(translated_content)
                    print(f"Translated file saved as: {translated_file_path}")



if __name__ == "__main__":
    folder_path = "./data/task_prompts" 
    target_languages = languages.keys()
    translate_markdown_files(folder_path, target_languages)