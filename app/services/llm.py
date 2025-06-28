import json
import os 
import httpx
import asyncio
from fastapi.responses import StreamingResponse


from langchain_core.messages import HumanMessage

from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from .memory import get_chat_memory
from ..utils import format_message
from dotenv import load_dotenv
from .memoryManager import MemoryManager
# load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json","Accept": "text/event-stream"}

summaryLLM = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=4000,
    streaming=False
)
def shouldSummarize(messages):
    """
    Determines if the conversation should be summarized based on the number of messages.
    """
    return len(messages)!=0 and len(messages)%6 == 0 # Example threshold, adjust as needed
async def streamLLMResponses(user_id: str, book_id: str, systemMessage: str, userMessage: str):
    memory = MemoryManager(user_id=user_id, book_id=book_id)
    redismemory = get_chat_memory(user_id=user_id, book_id=book_id)
    redismemory.save_message(userMessage, "user")
    allMessages = redismemory.get_messages()
    print(f"Messages in memory: {allMessages}")
    # Format the system and user messages

    if(shouldSummarize(allMessages)):
        print("Summarizing conversation...")
        lastSixConvos = allMessages[-6:]
        print(f"Last six conversations: {lastSixConvos}")
        summarySystemMessage = (
    "You are a memory compression assistant. Your job is to maintain a running summary of a conversation.\n\n"
    "- You will be given a previous summary (if any).\n"
    "- You will also be given the next few turns of conversation.\n"
    "- Your task is to return an updated summary that integrates the new information with the previous summary.\n"
    "- Be concise and preserve key context. Respond only with the updated summary. No explanations, no headings."
)   
        
            
        messagesToSummarize = [SystemMessage(content=summarySystemMessage)]
        previouSummary = redismemory.history.redis_client.get(f"summary:{user_id}:{book_id}")
        if previouSummary:
            previouSummary = previouSummary.decode('utf-8')
            messagesToSummarize.append(HumanMessage(content=f"Previous summary:\n{previouSummary}"))
        messagesToSummarize.append( HumanMessage(content="Here are the next few turns of the conversation:"))
        messagesToSummarize.extend(lastSixConvos)
        summary = await summaryLLM.ainvoke(messagesToSummarize)
        redismemory.history.redis_client.set(f"summary:{user_id}:{book_id}", summary, ex=7200)
          # Store summary for 1 hour


    formatted_messages = [{"role": "system", "content": systemMessage + "\n" +"Following is the conversation summary:\n" + summary }]

  
    formatted_messages.append({"role": "user", "content": userMessage})

    full_message = []  # ✅ Moved to outer scope
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
            print("Streaming complete, saving full message.")
            full_text = "".join(full_message)
            redismemory.save_message(full_text, "AI")
            print(f"Full message saved: {full_text}")
    return StreamingResponse(stream_response(), media_type="text/event-stream")
