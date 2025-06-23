from anthropic import Anthropic
from colorama import Fore, Style

from llms.base import LLM
from llms.utils import retry_with_exponential_backoff

class AnthropicLLM(LLM):
    def __init__(self, model_name: str) -> None:
        self.client = Anthropic()
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature=0.0,
        response_format=None,  # In order to comply to `run_simpleqa.py`
        **kwargs
    ) -> tuple[str, dict]:
        kwargs.pop("top_logprobs", None)  # top_logprobs is not available in Anthropic API
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            **kwargs
        )

        res_text = response.content[0].text
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage.input_tokens,  # Not available in Anthropic API
            "num_output_tokens": response.usage.output_tokens,  # Not available in Anthropic API
            "logprobs": []  # Not available in Anthropic API
        }
        return res_text, res_info

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature = 0.6,
        top_p = 0.9,
        **kwargs
    ) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=float(top_p),
            **kwargs
        )
        return response.content[0].text

if __name__ == "__main__":
    from pprint import pprint

    client = Anthropic()

    response = client.beta.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=12800,
        thinking={
            "type": "enabled",
            "budget_tokens": 4096
        },
        messages=[{
            "role": "user",
            "content": "Generate a comprehensive analysis of 1 + 2 + ... + 1000000"
        }],
        betas=["output-128k-2025-02-19"]
    )

    print(response)
