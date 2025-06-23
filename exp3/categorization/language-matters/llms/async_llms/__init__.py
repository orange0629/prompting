def get_async_llm(series: str, model_name: str):
    if series == "openai":
        from llms.async_llms.openai_llms import AsyncOpenAILLM
        return AsyncOpenAILLM(model_name)
    elif series == "google":
        from llms.async_llms.google_llms import AsyncGoogleLLM
        return AsyncGoogleLLM(model_name)
    elif series == "together":
        from llms.async_llms.together_llms import AsyncTogetherLLM
        return AsyncTogetherLLM(model_name)
    else:
        raise ValueError(f"Unsupported series: {series}")
