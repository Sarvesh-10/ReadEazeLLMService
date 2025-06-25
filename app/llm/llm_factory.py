import os
from langchain_groq import ChatGroq
# future: from langchain_openai import ChatOpenAI

from .model_enums import ModelProvider, ModelName

_llm_cache = {}

class LLMFactory:
    @staticmethod
    def get_llm(model: ModelName = ModelName.LLAMA3, provider: ModelProvider = ModelProvider.GROQ,callbacks=None):
        key = f"{provider.value}:{model.value}"

        if key in _llm_cache:
            print(f"[LLMFactory] Using cached LLM: {key}")
            return _llm_cache[key]

        if provider == ModelProvider.GROQ:
            print(f"[LLMFactory] Instantiating Groq LLM: {model.value}")
            llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=model.value,
                streaming=False,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        _llm_cache[key] = llm
        return llm
