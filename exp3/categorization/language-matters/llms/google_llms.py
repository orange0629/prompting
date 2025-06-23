import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from llms.base import LLM
from llms.utils import retry_with_exponential_backoff

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

class GoogleLLM(LLM):
    SAFETY_SETTINGS={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
    }

    def __init__(self, model_name: str) -> None:
        self.model = genai.GenerativeModel(model_name)

    @retry_with_exponential_backoff
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        **kwargs
    ) -> str:
        result = self.model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k)
            },
            safety_settings=self.SAFETY_SETTINGS,
            stream=False
        )
        res_text = result.candidates[0].content.parts[0].text
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": self.model.count_tokens(prompt).total_tokens,
            "num_output_tokens": self.model.count_tokens(res_text).total_tokens,
            "logprobs": []  # NOTE: currently the Google API does not provide logprobs
        }
        return res_text, res_info

if __name__ == "__main__":
    llm = GoogleLLM("gemini-1.5-pro")
    print(llm("Hi, what's your model name? Specifically, which version of Gemini is this?"))
