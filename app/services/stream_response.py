from fastapi.responses import StreamingResponse
from langchain.chains import ConversationChain
from langchain_core.messages import SystemMessage
from asyncio import Queue

from app.llm.llm_factory import LLMFactory
from app.services.chat_chain import get_conversation_chain
from ..configs.handlers import StreamingHandler
import app.llm.model_enums as enums
from .memoryManager import get_summary_memory

async def streamLLMResponses(
    user_id: str,
    book_id: str,
    userMessage: str,
    systemMessage: str,
    model: enums.ModelName = enums.ModelName.LLAMA3,
    provider: enums.ModelProvider = enums.ModelProvider.GROQ
):
    queue = Queue()
    handler = StreamingHandler(queue)

    # Create memory + attach handler-enabled LLM
    llm = LLMFactory.get_llm(model=model, provider=provider, callbacks=[handler])
    memory = get_summary_memory(user_id, book_id,llm=llm)
    # Optionally add system message at the start
    if systemMessage and not memory.chat_memory.messages:
        memory.chat_memory.messages.insert(0, SystemMessage(content=systemMessage))

    # Chain
    chain = get_conversation_chain(
        user_id=user_id,
        book_id=book_id,
        llm = llm
    )

    # Save user input
    memory.chat_memory.add_user_message(userMessage)

    async def token_stream():
        try:
            
            # Trigger LLM streaming
            async for _ in chain.astream({"input": userMessage}):
                pass
        except Exception as e:
            await queue.put(f"[ERROR] {str(e)}")
        finally:
            await queue.put("[END]")

        # Emit tokens pushed by handler
        while True:
            token = await queue.get()
            if token == "[END]":
                break
            if token.startswith("[ERROR]"):
                print(token)
                break
            yield token

    return StreamingResponse(token_stream(), media_type="text/event-stream")
