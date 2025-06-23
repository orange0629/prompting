import os
from together import Together
from openai import OpenAI
from llms.base import LLM
from llms.utils import retry_with_exponential_backoff

class TogetherLLM(LLM):
    TOP_LOGPROBS = 1  # currently only support 1

    def __init__(self, model_name="meta-llama/Llama-3-70b-chat-hf") -> None:
        # self.client = OpenAI(api_key=os.environ.get("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1")
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt, max_tokens=8192, temperature=0.0, **kwargs) -> str:
        if self.model_name[:len("deepseek-ai")] == "deepseek-ai":
            max_tokens = 32768
            if self.model_name == "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free":
                max_tokens = 6000
            temperature = 0.6
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=self.TOP_LOGPROBS,
            **kwargs
        )
        res_text = res.choices[0].message.content
        if res.choices[0].logprobs is not None:
            tokens = res.choices[0].logprobs.tokens
            logprobs = res.choices[0].logprobs.token_logprobs
        else:
            tokens = []
            logprobs = []
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": res.usage.prompt_tokens,
            "num_output_tokens": res.usage.completion_tokens,
            "logprobs": [{token: logprob} for token, logprob in zip(tokens, logprobs)]  # [{token: logit, ...}, ...]
        }
        return res_text, res_info

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        if self.model_name[:len("deepseek-ai")] == "deepseek-ai":
            max_tokens = 32768
            temperature = 0.6
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content

    # @retry_with_exponential_backoff
    def text_completion(
        self,
        prompt: str,
        max_tokens: int = 32768,
        temperature: float = 0.6,
        **kwargs
    ) -> str:
        res = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=self.TOP_LOGPROBS,
            **kwargs
        )
        res_text = res.choices[0].text
        tokens = res.choices[0].logprobs.tokens
        logprobs = res.choices[0].logprobs.token_logprobs
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": res.usage.prompt_tokens,
            "num_output_tokens": res.usage.completion_tokens,
            "logprobs": [{token: logprob} for token, logprob in zip(tokens, logprobs)]  # [{token: logit, ...}, ...]
        }
        return res_text, res_info

if __name__ == "__main__":
    from time import time
    start_time = time()
    llm = TogetherLLM(model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    res_text, res_info = llm.text_completion("1 + 1 = ?")
    print(res_text)
    print(res_info)
    print(f"Time taken: {time() - start_time} seconds")
