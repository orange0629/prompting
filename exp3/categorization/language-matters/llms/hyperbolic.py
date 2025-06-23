import os
import openai

from llms.base import LLM
from llms.utils import retry_with_exponential_backoff

class HyperbolicLLM(LLM):
    def __init__(self, model_name: str) -> None:
        self.client = openai.OpenAI(
            api_key=os.getenv("HYPERBOLIC_API_KEY"),
            base_url="https://api.hyperbolic.xyz/v1",
        )
        self.model_name = model_name

    @retry_with_exponential_backoff
    def __call__(self, prompt: str, **kwargs) -> tuple[str, dict]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        res_text = response.choices[0].message.content
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens,
            "logprobs": []
        }
        return res_text, res_info        

if __name__ == "__main__":
    from time import time
    start_time = time()
    llm = HyperbolicLLM(model_name="deepseek-ai/DeepSeek-R1")
    res_text, res_info = llm("Answer the following question based on the provided choices:\n\nQuestion: Enya and John are of normal phenotype but they have a son with a genetic condition that is caused by a mutation in the DMD gene and is a sex-linked recessive disorder. What are the chances of having a carrier and diseased child with each pregnancy if the sex of the child is unknown and there were no epigenetic changes involved in the progression of the disease?\n\nChoices:\nA. 50% carrier and 25% diseased\nB. 25% carrier and 25% diseased\nC. 50% carrier and 50% diseased\nD. 100% carrier and 50% diseased\n\nProvide your reasoning process first, then provide your final answer in the following format:\nANSWER: <letter>")
    print(res_text)
    print(res_info)
    print(f"Time taken: {time() - start_time} seconds")
