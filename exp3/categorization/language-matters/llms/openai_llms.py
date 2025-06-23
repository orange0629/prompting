import os
import openai
from openai import OpenAI
from pydantic import BaseModel
from colorama import Fore, Style

from llms.base import LLM
from llms.utils import retry_with_exponential_backoff

class OpenAILLM(LLM):
    TOP_LOGPROBS = 1

    def __init__(self, model_name: str) -> None:
        params = {'api_key': os.environ['OAI_KEY']}
        if os.getenv('CUSTOM_API_URL') and 'gpt-' not in model_name:
            params['base_url'] = os.environ['CUSTOM_API_URL']
            if 'CUSTOM_API_KEY' in os.environ:
                params['api_key'] = os.environ['CUSTOM_API_KEY']
        self.client = OpenAI(**params)
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature = 0.0,
        **kwargs
    ) -> tuple[str, dict]:
        if (kwargs.get("response_format", None) is not None) and issubclass(kwargs["response_format"], BaseModel):
            return self.call_with_json_schema(prompt, max_tokens, temperature, **kwargs)
        # Handle exceptions from reasoning models
        top_logprobs = kwargs.pop("top_logprobs", self.TOP_LOGPROBS)
        if self.model_name.startswith("o"):
            top_logprobs = None
            temperature = 1.0
            max_tokens = 25000
        if top_logprobs is None:
            return_logprobs = False
        else:
            return_logprobs = True
        return_logprobs = False
        kwargs['max_tokens'] = max_tokens
        
        messages = [{'role': 'user', 'content': prompt}]

        if 'deepseek' in self.model_name.lower():
            temperature = 0.6
            kwargs = {
                'top_p': 0.95,
            }
        if 'nemotron' in self.model_name.lower():
            temperature = 0.6
            kwargs = {
                'top_p': 0.95,
            }
            messages = [{"role": "system", "content": "detailed thinking on"}, 
                        {'role': 'user', 'content': prompt}]
        if 'qwen3' in self.model_name.lower():
            temperature = 0.6
            kwargs = {
                'top_p': 0.95,
            }
        if 'phi-4' in self.model_name.lower():
            if 'mini' in self.model_name.lower():
                temperature = 0.8
                kwargs = {
                    'top_p': 0.95,
                }
            else:                
                temperature = 0.8
                kwargs = {
                    'top_p': 0.95,
                    'extra_body': {'top_k': 50}
                }
            messages = [{"role": "system", "content": "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:"},
                        {'role': 'user', 'content': prompt}
                    ]

        response = self.client.chat.completions.create(
            model=self.model_name.lower(),
            messages=messages,
            temperature=float(temperature),
            logprobs=return_logprobs,
            **kwargs
        )
        log_prob_seq = []
        if return_logprobs:
            log_prob_seq = response.choices[0].logprobs.content
            assert abs(response.usage.completion_tokens - len(log_prob_seq)) <= 5
        res_text = response.choices[0].message.content
        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            print(response.choices[0].message.reasoning_content)
            res_text = '<think>'+response.choices[0].message.reasoning_content+'</think>'+res_text
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens,
            # "logprobs": [[{"token": pos_info.token, "logprob": pos_info.logprob} for pos_info in position.top_logprobs] for position in log_prob_seq]
        }
        return res_text, res_info

    def call_with_json_schema(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature=0.0,
        **kwargs
        ) -> tuple[str, dict]:
        assert issubclass(kwargs["response_format"], BaseModel)
        top_logprobs = kwargs.pop("top_logprobs", self.TOP_LOGPROBS)
        if top_logprobs is None:
            return_logprobs = False
        else:
            return_logprobs = True
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name.lower(),
                messages=[{'role': 'user', 'content': prompt}],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                logprobs=return_logprobs,
                top_logprobs=top_logprobs,
                **kwargs
            )
        except openai.LengthFinishReasonError:
            print(Fore.RED + "openai.LengthFinishReasonError" + Style.RESET_ALL)
            return "openai.LengthFinishReasonError", {
                "input": prompt,
                "output": "openai.LengthFinishReasonError",
                "num_input_tokens": len(prompt) // 4,
                "num_output_tokens": 0,
                "logprobs": []
            }
        log_prob_seq = response.choices[0].logprobs.content
        if log_prob_seq is None:
            print(Fore.YELLOW + f"log_prob_seq is None. response = {response}" + Style.RESET_ALL)
        else:
            assert response.usage.completion_tokens == len(log_prob_seq)
        res_text = response.choices[0].message.content
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens,
            "logprobs": [[{"token": pos_info.token, "logprob": pos_info.logprob} for pos_info in position.top_logprobs] for position in log_prob_seq] if (log_prob_seq is not None) else []
        }
        return res_text, res_info


    @retry_with_exponential_backoff
    def text_completion(
            self,
            prompt: str,
            max_tokens: int = 4096,
            temperature = 0.0,
            **kwargs
        ) -> tuple[str, dict]:
        top_logprobs = kwargs.pop("top_logprobs", self.TOP_LOGPROBS)
        if self.model_name.startswith("o"):
            top_logprobs = None
            temperature = 1.0
            max_tokens = 25000
        if 'deepseek' in self.model_name.lower():
            temperature = 0.6
            kwargs = {
                'top_p': 0.95,
            }
        if 'qwen3' in self.model_name.lower():
            temperature = 0.6
            kwargs = {
                'top_p': 0.95,
            }
        if 'phi-4' in self.model_name.lower():
            if 'mini' in self.model_name.lower():
                temperature = 0.8
                kwargs = {
                    'top_p': 0.95,
                }
            else:                
                temperature = 0.8
                kwargs = {
                    'top_p': 0.95,
                }
        if top_logprobs is None:
            return_logprobs = False
        else:
            return_logprobs = True
        response = self.client.completions.create(
            model=self.model_name.lower(),
            prompt=prompt,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            **kwargs
        )
        log_prob_seq = []
        res_text = response.choices[0].text
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens,
        }
        return res_text, res_info

    def generate(
            self,
            messages: list[dict],
            max_tokens: int = 4096,
            temperature = 0.0,
            seed: int = 42,
            **kwargs
        ) -> str:
        if 'qwen3' in self.model_name.lower():
            temperature = 0.6
            kwargs = {
                'top_p': 0.95,
            }
        if 'phi-4' in self.model_name.lower():
            if 'mini' in self.model_name.lower():
                temperature = 0.8
                kwargs = {
                    'top_p': 0.95,
                }
            else:                
                temperature = 0.8
                kwargs = {
                    'top_p': 0.95,
                }
        response = self.client.chat.completions.create(
            model=self.model_name.lower(),
            messages=messages,
            temperature=float(temperature),
            max_completion_tokens=int(max_tokens),
            seed=seed,
            **kwargs
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    llm = OpenAILLM( "qwen/qwen3-30b-a3b")
    otput, res = llm("<|im_start|>system<|im_sep|>\nYou are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:<|im_end|>\n<|im_start|>user<|im_sep|>\nWhat is the derivative of x^2?<|im_end|>\n<|im_start|>assistant<|im_sep|><think>")
    # llm = OpenAILLM( "nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
    # otput, res = llm("What is the derivative of x^2?")
    print(otput)
