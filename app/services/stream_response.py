import asyncio
from fastapi.responses import StreamingResponse
from langchain.chains import ConversationChain
from langchain_core.messages import SystemMessage
from asyncio import Queue
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.llm.llm_factory import LLMFactory
from app.services.chat_chain import get_conversation_chain
from ..configs.handlers import StreamingHandler
import app.llm.model_enums as enums
from langchain.chains import LLMChain
from .memoryManager import get_summary_memory
import logging

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

    # Create LLM with streaming handler
    llm = LLMFactory.get_llm(model=model, provider=provider, callbacks=[handler])

    # Get memory for session
    memory = get_summary_memory(user_id, book_id, llm=llm)

    # Define custom prompt with system message (not stored in memory)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=systemMessage),  # This affects LLM behavior but isn't saved
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # Create the LLMChain
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    response = await chain.ainvoke({"input": userMessage})
    return {"response": response.content}