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
    

    # Create LLM with streaming handler
    llm = LLMFactory.get_llm(model=model, provider=provider)

    # Get memory for session
    memory = get_summary_memory(user_id, book_id, llm=llm)

    # Define custom prompt with system message (not stored in memory)
    prompt = ChatPromptTemplate.from_template(
    """
    {system_message}

    Previous conversation summary:
    {history}

    User: {input}
    """
)


    # Create the LLMChain
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    response = await chain.ainvoke({"input": userMessage})
    print(f"Response from LLM: {response}")
    logging.info(f"Response from LLM: {response}")

    return {"response": response['text']}