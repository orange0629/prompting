import os, asyncio
from together import AsyncTogether

from llms.async_llms.utils import retry_with_exponential_backoff

class AsyncTogetherLLM:

    def __init__(self, model_name: str) -> None:
        self.client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
        self.model_name = model_name

    @retry_with_exponential_backoff
    async def __call__(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs
    )-> tuple[str, dict]:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        res_text = response.choices[0].message.content
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens,
        }
        return res_text, res_info

async def main():
    from tqdm.asyncio import tqdm

    llm = AsyncTogetherLLM(model_name="Qwen/QwQ-32B")
    prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Portugal?",
        "What is the capital of Greece?",
        "What is the capital of Turkey?",
    ]
    results = await tqdm.gather(
        *[llm(prompt) for prompt in prompts],
        desc="Generating responses",
        total=len(prompts),
    )
    for prompt, (res_text, res_info) in zip(prompts, results):
        print(f"Prompt: {prompt}")
        print(f"Response: {res_text}")
        print(f"Input tokens: {res_info['num_input_tokens']}")
        print(f"Output tokens: {res_info['num_output_tokens']}")
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())
