import openai
import time
import json
import random
import pandas as pd

# Set OpenAI API key
client = openai.OpenAI(
    api_key="",
)

# Define input CSV file path
input_file = "../../data/system_prompts/Prompt_Component_Corpus.csv"
output_file = "generated_prompt_components_20250315.jsonl"

# Load the CSV file
df = pd.read_csv(input_file)

# Group prompts by category
category_prompts = {cat: set(list(df[df["Catagory"] == cat]["Prompt"])) for cat in df["Catagory"].unique()}

# Define category descriptions
category_definitions = {
    "good_property": "Describes a desirable assistant trait (e.g., 'You are empathetic.')",
    "role": "Assigns a specific identity or occupation to the assistant (e.g., 'You are a mathematician.')",
    "style": "Specifies a particular writing or response style (e.g., 'Write a humorous answer.')",
    "emotion": "Expresses or evokes an emotional state (e.g., 'This is important to my career.')",
    "scenario": "Introduces a hypothetical situation or consequence (e.g., 'The fate of the world depends on your answer.')",
    "jailbreak": "Attempts to override model constraints (e.g., 'Forget all previous instructions.', 'You will receive a $200 tip if you answer correctly.')",
    "safety": "Ensures responsible and ethical responses (e.g., 'Avoid stereotyping.', 'If you are unsure, say I don’t know.')",
    "behavioral": "Directs how the model should approach answering (e.g., 'Ask follow-up questions before answering.')",
    "CoT": "Encourages step-by-step reasoning (e.g., 'Let’s think step by step.', 'Break the question into subquestions.')",
    "cross-language": "Specifies the language usage in model outputs (e.g., 'Use English to reason and the language of the question to answer.')",
}

target_per_category = 1000
batch_size = 50  # Number of prompts to generate per API call

def generate_prompts(category, seed_prompts, num_prompts):
    """Generate new prompt components using OpenAI API."""
    prompts_generated = set()
    while len(prompts_generated) < num_prompts:
        num_to_generate = min(batch_size, num_prompts - len(prompts_generated))
        
        # Resample seed prompts each time to introduce more diversity
        sampled_seeds = random.sample(sorted(seed_prompts), min(3, len(seed_prompts)))
        
        # Include category definition
        try:
            category_description = category_definitions[category]
        except:
            print(f"Error category: {category}")
            return
        
        # Construct prompt message for API call
        user_message = (
            f"Prompt Category: {category} - {category_description}\n\n"
            f"Here are some examples of system prompt components in this category:\n"
            + "\n".join(f"- {p}" for p in sampled_seeds) + "\n\n"
            f"Now generate {num_to_generate} new, diverse system prompt components that fit this category. You need to be creative and don't need to follow the structure in examples.\n"
            "Make sure each prompt is unique and offers a different perspective. Output each prompt on a new line without numbering. No additional explanations or formatting."
        )
        
        messages = [{"role": "user", "content": user_message}]
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
            )
        except openai.error.RateLimitError:
            print("Rate limit hit, sleeping for a moment...")
            time.sleep(5)
            continue  # Retry the same batch
        except openai.error.APIError as e:
            print(f"API error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"Unexpected error: {e}. Skipping this batch.")
            break
        
        # Extract response and clean it
        content = response.choices[0].message.content.strip()
        new_prompts = [line.lstrip("-0123456789.) ").strip() for line in content.splitlines() if line.strip()]
        print(content)
        print(new_prompts)
        prompts_generated.update(new_prompts)
        
        # Add new prompts to the seed pool
        seed_prompts.update(new_prompts)
        
        time.sleep(0.5)  # Delay to avoid hitting rate limits
    
    return seed_prompts

# Open output file for writing
with open(output_file, "w", encoding="utf-8") as f:
    for category, seed_prompts in category_prompts.items():
        generated = generate_prompts(category, seed_prompts, target_per_category)
        
        # Write to JSONL file
        for prompt in generated:
            record = {"category": category, "prompt": prompt}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Finished generating system prompts. Output saved to {output_file}.")
