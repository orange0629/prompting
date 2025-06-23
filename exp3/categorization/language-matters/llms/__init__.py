def get_llm(series: str, model_name: str):
    if series == "openai":
        from llms.openai_llms import OpenAILLM
        return OpenAILLM(model_name)
    elif series == "anthropic":
        from llms.anthropic_llms import AnthropicLLM
        return AnthropicLLM(model_name)
    elif series == "google":
        from llms.google_llms import GoogleLLM
        return GoogleLLM(model_name)
    elif series == "together":
        from llms.together_llms import TogetherLLM
        return TogetherLLM(model_name)
    elif series == "hyperbolic":
        from llms.hyperbolic import HyperbolicLLM
        return HyperbolicLLM(model_name)
    elif series == "vllm":
        from llms.vllms import vLLM
        return vLLM(model_name)
    else:
        raise ValueError(f"Unsupported LLM series: {series}")