from langchain.callbacks.base import AsyncCallbackHandler
from asyncio import Queue

class StreamingHandler(AsyncCallbackHandler):
    def __init__(self, queue: Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.queue.put(token)

    async def on_llm_end(self, response, **kwargs):
        await self.queue.put("[END]")

    async def on_llm_error(self, error, **kwargs):
        await self.queue.put(f"[ERROR] {str(error)}")
