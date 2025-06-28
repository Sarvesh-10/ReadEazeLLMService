from langchain_community.chat_message_histories import RedisChatMessageHistory
from ..config import redis_client
import os
from dotenv import load_dotenv
# load_dotenv()
memory_cache = {}


class ChatMemory: 
    def __init__(self,user_id:str,book_id:str):
        self.session_key = f"session:{user_id}:book:{book_id}"
        self.history = RedisChatMessageHistory(session_id=self.session_key,url=os.getenv("REDIS_URL"))

    def save_message(self,message:str,role:str):
        if role == "user":
            self.history.add_user_message(message)
        else:
            self.history.add_ai_message(message)
        redis_client.expire(self.session_key,7200)

    def get_messages(self):
        return self.history.messages

    def clear_session(self):
        redis_client.delete(self.session_key)
        memory_cache.pop(self.session_key,None)


def get_chat_memory(user_id:str,book_id:str):
    key = f"session:{user_id}:book:{book_id}"
    if key not in memory_cache:
        memory_cache[key] = ChatMemory(user_id=user_id,book_id=book_id)
    return memory_cache[key]