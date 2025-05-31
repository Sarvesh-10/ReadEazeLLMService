from langchain.memory import ConversationSummaryMemory
from langchain_groq import ChatGroq
import os

class MemoryManager:
    _instances = {}

    def __new__(cls, user_id: str, book_id: str, threshold: int = 10):
        key = f"{user_id}:{book_id}"
        if key not in cls._instances:
            instance = super(MemoryManager, cls).__new__(cls)
            cls._instances[key] = instance
            instance.__init__(user_id, book_id, threshold)
        return cls._instances[key]

    def __init__(self, user_id: str, book_id: str, threshold: int = 10):
        self.user_id = user_id
        self.book_id = book_id
        self.threshold = threshold
        self.llm = ChatGroq(
            model=os.getenv("GROQ_CHAT_MODEL"),
            api_key=os.getenv("GROQ_API_KEY"),
        )

        self.memory = ConversationSummaryMemory(
            memory_key=f"session:{user_id}:book:{book_id}",
            max_token_limit=1000,
            return_messages=True,
            input_key="input",
            output_key="output",
            llm=self.llm,
        )

    def add_message(self, message: str, assistant_response: str):
        self.memory.save_context(
    {"input":message},
    {"output":assistant_response}
)


        if len(self.memory.chat_memory.messages) >= self.threshold:
            self.summarize_memory()

    def summarize_memory(self):
        summary = self.memory.load_memory_variables({})
        messages = summary.get("messages", [])

        summarized_message = " ".join([msg.get("content", "") for msg in messages])

        self.memory.clear()
        self.memory.save_context(
            {"input": {"role": "user", "content": summarized_message}},
            {"output": {"role": "assistant", "content": summarized_message}}
        )

    def get_memory(self):
        return self.memory.load_memory_variables({})

    def clear_memory(self):
        self.memory.clear()
