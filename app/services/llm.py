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

async def streamLLMResponses(user_id: str, book_id: str, systemMessage: str, userMessage: str):
    memory = MemoryManager(user_id=user_id, book_id=book_id)
    redismemory = get_chat_memory(user_id=user_id, book_id=book_id)
    redismemory.save_message(userMessage, "user")

    memory_vars = memory.get_memory()
    history_str = memory_vars.get("history", "")

    formatted_messages = [{"role": "system", "content": systemMessage}]

    if history_str:
        formatted_messages.append({"role": "system", "content": history_str})  # ✅ Corrected role

    formatted_messages.append({"role": "user", "content": userMessage})

    full_message = []  # ✅ Moved to outer scope

    print(f"Previous memory history: {history_str}")
    print(f"Formatted messages: {formatted_messages}")

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
                                    yield content  # ✅ no need for f-string

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
