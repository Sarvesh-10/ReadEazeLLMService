from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from .memory import get_chat_memory
from .chat_chain import get_conversation_chain
import app.llm.model_enums as enums
async def streamLLMResponses(
    user_id: str,
    book_id: str,
    user_message: str,
    system_message: str,
    model: enums.ModelName = enums.ModelName.LLAMA3,
    provider: enums.ModelProvider = enums.ModelProvider.GROQ
):
    # 1. Redis store for full conversation history (UI purposes)
    redismemory = get_chat_memory(user_id=user_id, book_id=book_id)
    redismemory.save_message(user_message, "user")

    # 2. LangChain memory-aware conversation chain
    chain = get_conversation_chain(user_id, book_id, model, provider)

    # 3. Add system message manually if it's the first time
    # This depends on your memory logic (optional)
    if system_message:
        chain.memory.chat_memory.messages.insert(0, SystemMessage(content=system_message))

    full_output = []

    async def stream_response():
        try:
            async for chunk in chain.astream({"input": user_message}):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
                    full_output.append(chunk.content)
        except Exception as e:
            print(f"[StreamError] {e}")

        full_text = "".join(full_output)
        if full_text:
            redismemory.save_message(full_text, "AI")
            print(f"[Saved] Full AI message: {full_text}")

    return StreamingResponse(stream_response(), media_type="text/event-stream")
