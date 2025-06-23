from openai import OpenAI
from llms.base import LLM

class vLLM(LLM):

    def __init__(self, model_name: str) -> None:
        self.client = OpenAI(
            base_url="http://localhost:30002/v1",
            api_key="."  # NOTE: set a random string as the api key is necessary
        )
        self.model_name = model_name

    def __call__(self, prompt: str, max_tokens: int = 4096, temperature = 0.0, **kwargs) -> tuple[str, dict]:
        top_logprobs = kwargs.pop("top_logprobs", None)
        if top_logprobs is None:
            return_logprobs = False
        else:
            return_logprobs = True
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=return_logprobs,
            top_logprobs=top_logprobs,
            **kwargs
        )
        log_prob_seq = []
        if return_logprobs:
            log_prob_seq = response.choices[0].logprobs.content
            assert abs(response.usage.completion_tokens - len(log_prob_seq)) <= 5
        res_text = response.choices[0].message.content
        res_info = {
            "input": prompt,
            "output": res_text,
            "num_input_tokens": response.usage.prompt_tokens,
            "num_output_tokens": response.usage.completion_tokens,
            "logprobs": [[{"token": pos_info.token, "logprob": pos_info.logprob} for pos_info in position.top_logprobs] for position in log_prob_seq]
        }
        return res_text, res_info

if __name__ == "__main__":
    llm = vLLM("Qwen/Qwen2.5-Math-1.5B-Instruct")
    print(llm("Solve 1 + 1 = ?"))
