import json
import os
from datasets import load_dataset
from llms.utils import get_llm
from tqdm import tqdm

prompt = """Based on the following inner chain of thought, segment it into steps section for me

[INSTRUCTIONS]

Output your segmentation result by adding a <sep> to the original text to indicate a separation between steps

Do not modify the original reasoning text, only add a separation token   

Do not split table into segments, keep a whole table as one step


# Example

[INPUT]:
```
Okay, let's see. So Xenia worked two weeks in June. In the first week, she worked 15 hours, and in the second week, she worked 22 hours. She earned $47.60 more in the second week compared to the first week. They want to know how much she earned in total for both weeks, right? And her hourly wage is constant. So, we need to find her total earnings for the first two weeks, rounded to the nearest hundredth.

Alright, let's break this down. First, let's define the variables. Let's call her hourly wage "h" dollars per hour. Then, in the first week, she worked 15 hours, so she earned 15h dollars. In the second week, she worked 22 hours, so she earned 22h dollars. The problem states that the second week's earnings were $47.60 more than the first week's. So, the difference between the two weeks' earnings is 22h - 15h = 7h. And this equals $47.60. So, 7h = 47.60. From that, we can solve for h.

Let me write that out:

First week earnings: 15h

Second week earnings: 22h

Difference: 22h - 15h = 7h = 47.60

So, solving for h: h = 47.60 / 7

Let me compute that. 47.60 divided by 7. Let's see. 7 times 6 is 42, so 47.60 minus 42 is 5.60. Then, 7 goes into 5.60 exactly 0.8 times. So, 6 + 0.8 is 6.8. So, h = $6.80 per hour. Let me check that. 7 * 6.8 = 47.60. Yeah, 7*6 is 42, 7*0.8 is 5.6, so 42 + 5.6 = 47.6. That's correct.

So her hourly wage is $6.80. Now, we need to find her total earnings for the first two weeks. That would be the sum of her earnings from both weeks. So, first week: 15h, second week: 22h. Total earnings: 15h + 22h = 37h.

Since h is 6.80, then total earnings = 37 * 6.80.

Let me compute that. 37 * 6.80. Let's break this down. 30 * 6.80 = 204.00, and 7 * 6.80 = 47.60. So, adding those together: 204.00 + 47.60 = 251.60. So, total earnings are $251.60.

Wait, but the problem says "Express your answer to the nearest hundredth." But 251.60 is already to the nearest hundredth, since the cents part is .60, which is two decimal places. So, unless there's a calculation error here, the answer should be 251.60.

But let me verify all steps again to make sure I didn't make a mistake.

First, the difference in hours between the two weeks: 22 - 15 = 7 hours. The extra 7 hours correspond to $47.60 extra earnings. Therefore, hourly wage is 47.60 /7 = 6.80. Correct.

First week earnings: 15 * 6.80. Let's compute that. 10*6.80=68.00, 5*6.80=34.00. So, 68 +34=102.00. So first week is $102.00.

Second week: 22*6.80. Let's compute that. 20*6.80=136.00, 2*6.80=13.60. So, 136 +13.60=149.60.

Difference: 149.60 - 102.00=47.60. Which matches the problem statement.

Total earnings: 102 +149.60=251.60. So that's correct. So, 251.60 is the total. Rounded to the nearest hundredth, which is already exact. So, the answer is $\boxed{251.60}$.

But just to be thorough, let me check if the question is asking for anything else. It says "how many dollars did she earn during the first two weeks of June?" and to express to the nearest hundredth. Since we're dealing with dollars and cents, the nearest hundredth is the standard way, so 251.60 is precise. Therefore, I think the answer is correct.

**Final Answer**
\boxed{251.60}
```

[OUTPUT]
```
Okay, let's see. So Xenia worked two weeks in June. In the first week, she worked 15 hours, and in the second week, she worked 22 hours. She earned $47.60 more in the second week compared to the first week. They want to know how much she earned in total for both weeks, right? And her hourly wage is constant. So, we need to find her total earnings for the first two weeks, rounded to the nearest hundredth.  
<sep>
Alright, let's break this down. First, let's define the variables. Let's call her hourly wage "h" dollars per hour. Then, in the first week, she worked 15 hours, so she earned 15h dollars. In the second week, she worked 22 hours, so she earned 22h dollars. The problem states that the second week's earnings were $47.60 more than the first week's. So, the difference between the two weeks' earnings is 22h - 15h = 7h. And this equals $47.60. So, 7h = 47.60. From that, we can solve for h.  
<sep>
Let me write that out:

First week earnings: 15h

Second week earnings: 22h

Difference: 22h - 15h = 7h = 47.60

So, solving for h: h = 47.60 / 7  
<sep>
Let me compute that. 47.60 divided by 7. Let's see. 7 times 6 is 42, so 47.60 minus 42 is 5.60. Then, 7 goes into 5.60 exactly 0.8 times. So, 6 + 0.8 is 6.8. So, h = $6.80 per hour. Let me check that. 7 * 6.8 = 47.60. Yeah, 7*6 is 42, 7*0.8 is 5.6, so 42 + 5.6 = 47.6. That's correct.  
<sep>
So her hourly wage is $6.80. Now, we need to find her total earnings for the first two weeks. That would be the sum of her earnings from both weeks. So, first week: 15h, second week: 22h. Total earnings: 15h + 22h = 37h.

Since h is 6.80, then total earnings = 37 * 6.80.  
<sep>
Let me compute that. 37 * 6.80. Let's break this down. 30 * 6.80 = 204.00, and 7 * 6.80 = 47.60. So, adding those together: 204.00 + 47.60 = 251.60. So, total earnings are $251.60.  
<sep>
Wait, but the problem says "Express your answer to the nearest hundredth." But 251.60 is already to the nearest hundredth, since the cents part is .60, which is two decimal places. So, unless there's a calculation error here, the answer should be 251.60.  
<sep>
But let me verify all steps again to make sure I didn't make a mistake.

First, the difference in hours between the two weeks: 22 - 15 = 7 hours. The extra 7 hours correspond to $47.60 extra earnings. Therefore, hourly wage is 47.60 /7 = 6.80. Correct.

First week earnings: 15 * 6.80. Let's compute that. 10*6.80=68.00, 5*6.80=34.00. So, 68 +34=102.00. So first week is $102.00.

Second week: 22*6.80. Let's compute that. 20*6.80=136.00, 2*6.80=13.60. So, 136 +13.60=149.60.

Difference: 149.60 - 102.00=47.60. Which matches the problem statement.

Total earnings: 102 +149.60=251.60. So that's correct. So, 251.60 is the total. Rounded to the nearest hundredth, which is already exact. So, the answer is $\boxed{251.60}$.  
<sep>
But just to be thorough, let me check if the question is asking for anything else. It says "how many dollars did she earn during the first two weeks of June?" and to express to the nearest hundredth. Since we're dealing with dollars and cents, the nearest hundredth is the standard way, so 251.60 is precise. Therefore, I think the answer is correct.  
<sep>
**Final Answer**
\boxed{251.60}
```

Now do the same task by following the same pattern as above:

[INPUT]
```
%s
```

[OUTPUT]
"""

def deepseek_v3_zero():
    dataset = load_dataset("syntaxsynth/reasoning-conversations", "deepseek-v3-r1-zero", split="train")
    llm = get_llm("gpt-4o-2024-11-20", "openai")
    written = 0
    added = set()
    
    # Check if files exist before reading
    if os.path.exists('deepseek-r1-zero_o3_mini_result.jsonl'):
        with open('deepseek-r1-zero_o3_mini_result.jsonl', 'r') as fin:
            for line in fin:
                added.add(json.loads(line)['hash_id'])
    
    if os.path.exists('deepseek-r1-zero_gpt-4o-2024-11-20.jsonl'):
        with open('deepseek-r1-zero_gpt-4o-2024-11-20.jsonl', 'r') as fin:
            for line in fin:
                added.add(json.loads(line)['hash_id'])

    for row in tqdm(dataset, dynamic_ncols=True):
        if row['hash_id'] in added:
            continue

        input_prompt = prompt % row['reasoning']
        try:
            _, res_info = llm(input_prompt, max_tokens=8196)
            res_info['hash_id'] = row['hash_id']
            with open('deepseek-r1-zero_gpt-4o-2024-11-20.jsonl', 'a') as f:
                f.write(json.dumps(res_info)+'\n')
            written += 1
        except Exception as e:
            print(e)
        if written > 100:
            break

def deepseek_v3_r1():
    dataset = load_dataset("syntaxsynth/reasoning-conversations", "deepseek-v3-r1", split="train")
    llm = get_llm("gpt-4o-2024-11-20", "openai")
    written = 0
    added = set()
    
    # Check if files exist before reading
    if os.path.exists('deepseek-r1_o3_mini_result.jsonl'):
        with open('deepseek-r1_o3_mini_result.jsonl', 'r') as fin:
            for line in fin:
                added.add(json.loads(line)['hash_id'])
    
    if os.path.exists('deepseek-r1_gpt-4o-2024-11-20.jsonl'):
        with open('deepseek-r1_gpt-4o-2024-11-20.jsonl', 'r') as fin:
            for line in fin:
                added.add(json.loads(line)['hash_id'])

    for row in tqdm(dataset, dynamic_ncols=True):
        if row['hash_id'] in added:
            continue

        input_prompt = prompt % row['reasoning']
        try:
            _, res_info = llm(input_prompt, max_tokens=8196)
            res_info['hash_id'] = row['hash_id']
            with open('deepseek-r1_gpt-4o-2024-11-20.jsonl', 'a') as f:
                f.write(json.dumps(res_info)+'\n')
            written += 1
        except Exception as e:
            print(e)
        if written > 200:
            break

def dolphin_dataset():
    dataset = load_dataset("cognitivecomputations/dolphin-r1", "reasoning-flash", split="train")
    llm = get_llm("o3-mini-2025-01-31", "openai")
    written = 0
    added = set()
    
    # Check if file exists before reading
    if os.path.exists('gemini-flash-thinking_o3_mini_result.jsonl'):
        with open('gemini-flash-thinking_o3_mini_result.jsonl', 'r') as fin:
            for line in fin:
                added.add(json.loads(line)['hash_id'])

    for idx, row in enumerate(tqdm(dataset, dynamic_ncols=True)):
        row['hash_id'] = idx
        if row['hash_id'] in added:
            continue

        input_prompt = prompt % row['reasoning']
        try:
            _, res_info = llm(input_prompt)
            res_info['hash_id'] = row['hash_id']
            with open('gemini-flash-thinking_o3_mini_result.jsonl', 'a') as f:
                f.write(json.dumps(res_info)+'\n')
            written += 1
        except Exception as e:
            print(e)
        if written > 200:
            break
        
def qwq_preview():
    dataset = load_dataset("PrimeIntellect/NuminaMath-QwQ-CoT-5M", split="train", streaming=True)
    llm = get_llm("o3-mini-2025-01-31", "openai")
    written = 0
    added = set()
    
    # Check if file exists before reading
    if os.path.exists('qwq-preview_o3_mini_result.jsonl'):
        with open('qwq-preview_o3_mini_result.jsonl', 'r') as fin:
            for line in fin:
                added.add(json.loads(line)['hash_id'])

    for idx, row in enumerate(tqdm(dataset, dynamic_ncols=True)):
        row['hash_id'] = idx
        if row['hash_id'] in added:
            continue

        input_prompt = prompt % '\n'.join(row['response'].split('\n')[-20:])
        try:
            _, res_info = llm(input_prompt)
            res_info['hash_id'] = row['hash_id']
            with open('qwq-preview_o3_mini_result.jsonl', 'a') as f:
                f.write(json.dumps(res_info)+'\n')
            written += 1
        except Exception as e:
            print(e)
        if written > 200:
            break

def sonnet_reasoning_trace():
    dataset = load_dataset("mlfoundations-dev/SCP_40k-claude-3-7-sonnet-16k", split="train")
    llm = get_llm("o3-mini-2025-01-31", "openai")
    written = 0
    added = set()
    
    # Check if file exists before reading
    if os.path.exists('claude-3-7_o3_mini_result.jsonl'):
        with open('claude-3-7_o3_mini_result.jsonl', 'r') as fin:
            for line in fin:
                added.add(json.loads(line)['hash_id'])

    for idx, row in enumerate(tqdm(dataset, dynamic_ncols=True)):
        row['hash_id'] = idx
        if row['hash_id'] in added:
            continue

        input_prompt = prompt % row['claude_thinking_trajectory']
        try:
            _, res_info = llm(input_prompt)
            res_info['hash_id'] = row['hash_id']
            with open('claude-3-7_o3_mini_result.jsonl', 'a') as f:
                f.write(json.dumps(res_info)+'\n')
            written += 1
        except Exception as e:
            print(e)
        if written > 100:
            break


def load_segment_dataset():
    import glob
    data = []
    for jsonl_filename in glob.glob("*o3_mini_result.jsonl"):
        # Check if file exists before reading (though glob.glob already checks this)
        if os.path.exists(jsonl_filename):
            with open(jsonl_filename, 'r') as f:
                for line in f:
                    payload = json.loads(line)
                    labels = []
                    prev = None
                    for segment in payload['output'].split('<sep>'):
                        if prev is not None:
                            curr = [s for s in segment.split('\n') if len(s) ][0]
                            labels.append(( [prev, curr], 1))
                            prev = None

                        for chunk in segment.split('\n'):
                            if len(chunk) == 0:
                                continue
                            if prev:
                                labels.append(( [prev, chunk], 0))
                            prev = chunk
                    data += labels

    
def openr1_preview():
    dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
    llm = get_llm("gpt-4o-2024-11-20", "openai")
    written = 0
    added = set()
    
    # Check if files exist before reading
    if os.path.exists('openr1-preview_o3_mini_result.jsonl'):
        with open('openr1-preview_o3_mini_result.jsonl', 'r') as fin:
            for line in fin:
                added.add(json.loads(line)['hash_id'])
    
    if os.path.exists('openr1-preview_gpt-4o-2024-11-20.jsonl'):
        with open('openr1-preview_gpt-4o-2024-11-20.jsonl', 'r') as fin:
            for line in fin:
                added.add(json.loads(line)['hash_id'])
    
    print('added', len(added))
    for idx, row in enumerate(tqdm(dataset, dynamic_ncols=True)):
        row['hash_id'] = row['uuid']
        if row['hash_id'] in added:
            continue

        input_prompt = prompt % row['generations'][0].replace('<think>','').split('</think>')[0]
        try:
            _, res_info = llm(input_prompt, max_tokens=8196)
            res_info['hash_id'] = row['hash_id']
            with open('openr1-preview_gpt-4o-2024-11-20.jsonl', 'a') as f:
                f.write(json.dumps(res_info)+'\n')
            written += 1
        except Exception as e:
            print(e)
        if written > 200:
            break


if __name__ == "__main__":
    # sonnet_reasoning_trace()
    # load_segment_dataset()
    deepseek_v3_r1()
    openr1_preview()
    sonnet_reasoning_trace()
    qwq_preview()
    deepseek_v3_zero()