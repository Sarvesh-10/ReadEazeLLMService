# chat_chain.py

from langchain.chains import ConversationChain
from app.llm.model_enums import ModelName, ModelProvider
from .memoryManager import get_summary_memory
# Cache to store chains per user-book session
_chain_cache = {}

def get_conversation_chain(user_id: str, book_id: str, llm):
    session_key = f"{user_id}:{book_id}"
    
    if session_key not in _chain_cache:
        memory = get_summary_memory(user_id, book_id,llm=llm)
        chain = ConversationChain(
            llm=llm,  # LLM already available from memory
            memory=memory,
            verbose=False,
        )
        _chain_cache[session_key] = chain
        print(f"[ChatChain] Created new chain for session: {session_key}")
    else:
        print(f"[ChatChain] Using cached chain for session: {session_key}")

    return _chain_cache[session_key]
