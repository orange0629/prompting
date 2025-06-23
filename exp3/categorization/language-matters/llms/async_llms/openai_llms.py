from openai import AsyncOpenAI

from llms.async_llms.utils import retry_with_exponential_backoff

class AsyncOpenAILLM:
    def __init__(self, model_name: str) -> None:
        self.client = AsyncOpenAI()
        self.model_name = model_name

    @retry_with_exponential_backoff
    async def __call__(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs
    ) -> tuple[str, dict]:
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
