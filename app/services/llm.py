import json
import os 
import httpx
import asyncio
from fastapi.responses import StreamingResponse


from langchain_core.messages import HumanMessage

from langchain_core.messages import AIMessage

from langchain_core.messages import SystemMessage
from .memory import get_chat_memory
from ..utils import format_message
from dotenv import load_dotenv
from .memoryManager import MemoryManager
# load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json","Accept": "text/event-stream"}

async def streamLLMResponses(user_id:str,book_id:str,systemMessage: str, userMessage: str):
    memory = MemoryManager(user_id=user_id, book_id=book_id)
    redismemory = get_chat_memory(user_id=user_id, book_id=book_id)
    redismemory.save_message(userMessage, "user")
    previous_messages = memory.get_memory()
    print(f"Previous messages: {previous_messages}")
    session_messages = previous_messages.get(f"session:{user_id}:book:{book_id}", [])
    formatted_messages = []
    for msg in session_messages:
        role = "assistant" if isinstance(msg, SystemMessage) else "user"
        content = msg.content
        formatted_messages.append({"role": role, "content": content})
    full_message = []
    formatted_messages.extend([
    {"role": "system", "content": systemMessage},
    {"role": "user", "content": userMessage}
])
    async def stream_response():
        print("Sending request to Groq...")
        payload = {
            "model": "llama3-8b-8192",
            "messages": formatted_messages,
            "stream": True
        }

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", GROQ_API_URL, headers=HEADERS, json=payload) as response:
                    print(f"Groq response status: {response.status_code}")
                    print(f"Groq headers: {response.headers}")

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue

                        print(f"RAW line: {line}")

                        if line == "data: [DONE]":
                            break

                        if line.startswith("data: "):
                            line = line[6:].strip()

                        try:
                            parsed = json.loads(line)
                            choices = parsed.get("choices", [])
                            if choices and "delta" in choices[0]:
                                content = choices[0]["delta"].get("content", "")
                                if content:
                                    print(f"Streaming content: {content}")
                                    full_message.append(content)
                                    yield f"{content}"
                                    # await asyncio.sleep(0)

                        except json.JSONDecodeError as e:
                            print(f"Malformed JSON: {line} - Error: {e}")
                            continue

        except Exception as e:
            print(f"Error during streaming: {e}")
        if full_message:
            full_text = "".join(full_message)
            redismemory.save_message(full_text, "AI")
            memory.add_message(userMessage, full_text)

    return StreamingResponse(stream_response(), media_type="text/event-stream")