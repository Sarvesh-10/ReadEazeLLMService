import json
import os
from typing import Optional 
import httpx
from fastapi.responses import StreamingResponse


from langchain_core.messages import HumanMessage

from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from qdrant_client import QdrantClient
from .memory import get_chat_memory
from ..utils import format_message
# from dotenv import load_dotenv
from .memoryManager import MemoryManager
from ..customLogging import logger
from qdrant_client.http import models as qdrant_models

# load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json","Accept": "text/event-stream"}
QDRANT_URL = "http://qdrant:6333"
QDRANT_COLLECTION = "books_collection"
HF_EMBEDDING_URL = os.getenv("HF_EMBEDDING_URL", "http://huggingface-embeddings:80/predict")
API_URL = 'https://sarvesh92-sentence-transformers-all-minilm-l6-v2.hf.space/embed'

summaryLLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=4000,
    streaming=False
)
client = QdrantClient(QDRANT_URL)

def getContextFromQdrant(query: str, user_id: str, book_id: str, top_k: int = 3):
    try:
        # Get embeddings for the query
        response = httpx.post(API_URL, json={"texts": [query]})
        response.raise_for_status()
        query_embedding = response.json()['embedding'][0]

        # Initialize Qdrant client

        # Build filter for user_id and book_id
        qdrant_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="user_id",
                    match=qdrant_models.MatchValue(value=user_id)
                ),
                qdrant_models.FieldCondition(
                    key="book_id",
                    match=qdrant_models.MatchValue(value=book_id)
                ),
            ]
        )

        # Perform similarity search with filter
        search_result = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            query_filter=qdrant_filter
        )

        # Extract and return the relevant text chunks
        context_chunks = []
        for point in search_result:
            payload = point.payload
            # Try 'text', fallback to 'chunk' for compatibility
            if payload:
                if 'text' in payload:
                    context_chunks.append(payload['text'])
                elif 'chunk' in payload:
                    context_chunks.append(payload['chunk'])

        return context_chunks

    except httpx.HTTPError as e:
        logger.error(f"HTTP error during embedding retrieval: {e}")
    except Exception as e:
        logger.error(f"Error during Qdrant search: {e}")

    return []
def buildConversationContext(messages: list):

    "write the code to build the conversation context by taking last len(messages) % mod messages from the list of messages"

    logger.info("Building conversation context from messages.")
    if len(messages) == 0:
        logger.warning("No messages found in conversation history.")
        return []
    count = len(messages) % 6
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
    

    logger.info(f"Streaming LLM responses for user: {user_id}, book: {book_id}")
    logger.info(f"System message: {systemMessage}")
    logger.info(f"User message: {userMessage}")
    logger.info(f"Total messages in history: {len(allMessages)}")
    logger.info(f"All messages: {allMessages}")
    previouSummary = redismemory.history.redis_client.get(f"summary:{user_id}:{book_id}")
    if previouSummary:
        previouSummary = previouSummary.decode('utf-8')
        logger.info(f"Previous summary found: {previouSummary}")
        systemMessage = f"{systemMessage}\n\nHere is the previous summary of the conversation:\n{previouSummary}"
    # Format the system and user messages
    if(shouldSummarize(allMessages)):
        logger.info("Summarization needed, processing last six messages.")
        lastSixConvos = allMessages[-6:]
        print(f"Last six conversations: {lastSixConvos}")
        summarySystemMessage = """Summarize the conversation below, combining it with any previous summary. Keep it short and focused. Only return the updated summary — no explanation.
        This is a conversation and summary between a user and an AI assistant. Please summarize this in a neutral tone, without any personal opinions or biases. The summary should be concise and to the point, capturing the main ideas and key points discussed in the conversation."""
        
            
        messagesToSummarize = [SystemMessage(content=summarySystemMessage)]
        previouSummary = redismemory.history.redis_client.get(f"summary:{user_id}:{book_id}")
        if previouSummary:
            previouSummary = previouSummary.decode('utf-8')
            logger.info(f"Previous summary found: {previouSummary}")
            prevMemoryIncluded = True
            messagesToSummarize.append(HumanMessage(content=f"Previous summary:\n{previouSummary}"))
        messagesToSummarize.append( HumanMessage(content="Here are the next few turns of the conversation:"))
        messagesToSummarize.extend(lastSixConvos)
        messagesToSummarize.append(HumanMessage(content="Please summarize the conversation so far.I hav also given the previous summary if it exists"))
        summary = await summaryLLM.ainvoke(messagesToSummarize)
        logger.info(f"Generated summary: {summary.content.strip()}")
        redismemory.history.redis_client.set(f"summary:{user_id}:{book_id}", summary.content.strip(), ex=7200)
        systemMessage = f"{systemMessage}\n\nHere is the updated summary of the conversation:\n{summary.content.strip()}"
          # Store summary for 1 hour


        
        
    
    formatted_messages = buildConversationContext(allMessages)
    formatted_messages.append({"role": "system", "content": systemMessage})
    formatted_messages.append({"role": "user", "content": userMessage})
    
    context = getContextFromQdrant(userMessage, user_id, book_id, top_k=5)
    logger.info(f"Retrieved {len(context)} context chunks from Qdrant.")
    logger.info(f"Context chunks: {context}")
    if context:
        context_str = "\n\n".join(context)
        formatted_messages.insert(-2, {"role": "system", "content": f"Here are some relevant excerpts from the book to help you answer the user's question:\n{context_str}"})
        logger.info("Added context from Qdrant to the conversation.")
    else:
        logger.info("No relevant context found in Qdrant.")

    full_message = []  # ✅ Moved to outer scope


    async def stream_response():
        print("Sending request to Groq...")
        payload = {
            "model": "llama-3.3-70b-versatile",
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
            redismemory.save_message(userMessage, "user")
            redismemory.save_message(full_text, "AI")

            logger.info(f"Full message saved: {full_text}")
    return StreamingResponse(stream_response(), media_type="text/event-stream")
