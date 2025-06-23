import os
from google import genai
from google.genai import types

from llms.async_llms.utils import retry_with_exponential_backoff

class AsyncGoogleLLM:
    def __init__(self, model_name: str) -> None:
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model_name = model_name

    @retry_with_exponential_backoff
    async def __call__(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        thinking_budget: int = 0,
        **kwargs
    ) -> tuple[str, dict]:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget,
            ),
            response_mime_type="text/plain"
        )
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=generate_content_config
        )
        res_text = response.text
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage_metadata.prompt_token_count,
            "num_output_tokens": response.usage_metadata.candidates_token_count,
        }
        return res_text, res_info

async def main():
    llm = AsyncGoogleLLM(model_name="gemini-2.5-flash-preview-04-17")
    res_text, res_info = await llm("Say 1 one hundred times without newlines.")
    print(res_text)
    print(res_info)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
