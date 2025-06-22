# memory_manager.py

from langchain.memory import ConversationSummaryMemory
import app.llm.llm_factory as factory
import app.llm.model_enums as enums

# Cache to store memory per user-book session
_memory_cache = {}

def get_summary_memory(user_id: str, book_id: str, model: enums.ModelName = enums.ModelName.LLAMA3, provider: enums.ModelProvider = enums.ModelProvider.GROQ):
    session_key = f"{user_id}:{book_id}"
    
    if session_key not in _memory_cache:
        llm = factory.LLMFactory.get_llm(model=model, provider=provider)
        memory = ConversationSummaryMemory(
            llm=llm,
            memory_key="history",
            return_messages=True
        )
        _memory_cache[session_key] = memory
        print(f"[MemoryManager] Created memory for session: {session_key}")
    else:
        print(f"[MemoryManager] Using cached memory for session: {session_key}")

    return _memory_cache[session_key]
