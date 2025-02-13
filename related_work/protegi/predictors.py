import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
from lib.modelloader import inference_model


from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template

import utils
import tasks

class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass

class BinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        response = utils.chatgpt(
            prompt, max_tokens=4, n=1, timeout=2, 
            temperature=self.opt['temperature'])[0]
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred
    

class VLLMPredictor(GPT4Predictor):
    def __init__(self, model_dir, multi_thread=1, system_prompt=""):
        self.model_obj = inference_model(model_dir=model_dir, cache_dir="/shared/4/models/", multi_thread=multi_thread)
        self.system_prompt = system_prompt

    def inference(self, ex_lst, prompt):
        # user_prompt=Template(prompt).render(text=ex['text'])
        print(prompt)
        prompt_lst = [self.model_obj.get_prompt_template().format(system_prompt=self.system_prompt, user_prompt=str(prompt).replace("{question_prompt}", ex['text'])) for ex in ex_lst]
        response = self.model_obj.generate(prompt_lst, max_token_len=512, use_tqdm=True)
        print(response[:5])
        return response
