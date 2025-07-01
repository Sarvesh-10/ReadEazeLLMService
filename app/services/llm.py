import json
import os
from typing import Optional 
import httpx
from fastapi.responses import StreamingResponse


from langchain_core.messages import HumanMessage

from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from .memory import get_chat_memory
from ..utils import format_message
# from dotenv import load_dotenv
from .memoryManager import MemoryManager
from ..customLogging import logger

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
def buildConversationContext(messages: list,mod:int):

    "write the code to build the conversation context by taking last len(messages) % mod messages from the list of messages"

    logger.info(f"Building conversation context with mod: {mod}")
    if len(messages) == 0:
        logger.warning("No messages found in conversation history.")
        return []
    count = len(messages) % mod
    if count == 0:
        logger.info("No messages to include in conversation context, returning empty list.")
        return []
    context = [format_message(msg) for msg in messages[-count:]]
    if not context:
        logger.warning("No messages to include in conversation context.")
        return []
    logger.info(f"Built conversation context with {len(context)} messages.")
    logger.info(f"Context: {context}")
    return context
    # Sort messages by timestamp if they have a 'timestamp' field
    
def shouldSummarize(messages):
    """
    Determines if the conversation should be summarized based on the number of messages.
    """
    print("Checking if summarization is needed...")
    
    print(f"Total messages: {len(messages)}")
    print("Checking if len(messages) % 6 == 0",len(messages) % 6 == 0)
    return len(messages)!=0 and len(messages)%6 == 0 # Example threshold, adjust as needed
async def streamLLMResponses(user_id: str, book_id: str, systemMessage: str, userMessage: str):
    memory = MemoryManager(user_id=user_id, book_id=book_id)
    redismemory = get_chat_memory(user_id=user_id, book_id=book_id)
    allMessages = redismemory.get_messages()
    prevMemoryIncluded = False
    

    logger.info(f"Streaming LLM responses for user: {user_id}, book: {book_id}")
    logger.info(f"System message: {systemMessage}")
    logger.info(f"User message: {userMessage}")
    logger.info(f"Total messages in history: {len(allMessages)}")
    logger.info(f"All messages: {allMessages}")
    # Format the system and user messages
    if(shouldSummarize(allMessages)):
        logger.info("Summarization needed, processing last six messages.")
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
            logger.info(f"Previous summary found: {previouSummary}")
            prevMemoryIncluded = True
            messagesToSummarize.append(HumanMessage(content=f"Previous summary:\n{previouSummary}"))
        messagesToSummarize.append( HumanMessage(content="Here are the next few turns of the conversation:"))
        messagesToSummarize.extend(lastSixConvos)
        summary = await summaryLLM.ainvoke(messagesToSummarize)
        logger.info(f"Generated summary: {summary.content.strip()}")
        redismemory.history.redis_client.set(f"summary:{user_id}:{book_id}", summary.content.strip(), ex=7200)
        systemMessage = f"{systemMessage}\n\nHere is the updated summary of the conversation:\n{summary.content.strip()}"
          # Store summary for 1 hour

    if not prevMemoryIncluded:
        previouSummary = redismemory.history.redis_client.get(f"summary:{user_id}:{book_id}")
        if previouSummary:
            previouSummary = previouSummary.decode('utf-8')
            logger.info(f"Previous summary found: {previouSummary}")
            prevMemoryIncluded = True
            systemMessage = f"{systemMessage}\n\nHere is the previous summary of the conversation:\n{previouSummary}"
        
    mod = len(allMessages)-len(allMessages)%6 + 2 if len(allMessages) > 6 else 6
    formatted_messages = buildConversationContext(allMessages,mod)
    formatted_messages.append({"role": "system", "content": systemMessage})
    formatted_messages.append({"role": "user", "content": userMessage})


    full_message = []  # ✅ Moved to outer scope


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
                    logger.info(f"Groq response status: {response.status_code}")
                    logger.info(f"Groq response headers: {response.headers}")

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue


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
                                    logger.info(f"Streaming content: {content}")
                                    full_message.append(content)
                                    yield content  # ✅ no need for f-string

                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e} for line: {line}")
                            continue

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            

        if full_message:
            logger.info("Streaming completed, saving full message.")
            full_text = "".join(full_message)
            redismemory.save_message(full_text, "AI")
            redismemory.save_message(userMessage, "user")

            logger.info(f"Full message saved: {full_text}")
    return StreamingResponse(stream_response(), media_type="text/event-stream")
