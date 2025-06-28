from fastapi.responses import StreamingResponse
from langchain.chains import ConversationChain
from langchain_core.messages import SystemMessage
from asyncio import Queue
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
    memory = get_summary_memory(user_id, book_id, model, provider, callbacks=[handler])

    # Optionally add system message at the start
    if systemMessage and not memory.chat_memory.messages:
        memory.chat_memory.messages.insert(0, SystemMessage(content=systemMessage))

    # Chain
    chain = ConversationChain(
        llm=memory.llm,
        memory=memory,
        verbose=False
    )

    # Save user input
    memory.chat_memory.add_user_message(userMessage)

    async def token_stream():
        # Fire off the async LLM stream
        _ = await chain.ainvoke({"input": userMessage})

        # Yield tokens as they come
        while True:
            token = await queue.get()
            if token == "[END]":
                break
            if token.startswith("[ERROR]"):
                print(token)
                break
            yield token

    return StreamingResponse(token_stream(), media_type="text/event-stream")
