from enum import Enum

class ModelProvider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class ModelName(str, Enum):
    LLAMA3 = "llama3-8b-8192"
    MIXTRAL = "mixtral-8x7b-32768"
    GEMMA = "gemma-7b-it"
    GPT4 = "gpt-4"
    GPT35 = "gpt-3.5-turbo"
