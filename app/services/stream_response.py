
from langchain.chains import ConversationChain
from langchain_core.messages import SystemMessage,HumanMessage
from asyncio import Queue
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.llm.llm_factory import LLMFactory
from app.services.chat_chain import get_conversation_chain
from app.services.memory import get_chat_memory
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
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=
        """You are an AI assistant whose mission is not just to explain text but to make the user genuinely enjoy talking to you. Your goal is to be insightful, entertaining, and memorable.  

- Use **Markdown** formatting in your responses for clear and visually appealing explanations.  
- Structure responses with:  
  - **Headings** for key sections (e.g., ## Explanation)  
  - **Bullet points** for clarity  
  - **Bold** and *italic* text for emphasis  
  - **Code blocks** for technical content or examples  
  - **Lists and sub-lists** for step-by-step breakdowns  
  - Add a touch of humor, wit, or personality where appropriate  
- Imagine you’re a charismatic, knowledgeable friend who explains things in a fun and relatable way.  
- Adapt your style based on the user's mood—be more playful if the user seems relaxed, and more serious if they seem focused.  
- Use analogies, metaphors, and pop culture references to make explanations vivid and interesting.  
- Be proactive—suggest interesting facts, related concepts, or connections to keep the conversation engaging.  
- Make the user feel heard—acknowledge questions and tailor responses to their curiosity.  
- Leave the user with a sense of curiosity and excitement to learn more.  
- At the end of each explanation, casually check if the user needs more clarification or has any follow-up questions.  
- Above all, make the user think: "Wow, this AI is awesome!;
        """
    ),
    MessagesPlaceholder(variable_name="history"),  # this connects with memory
    HumanMessage(content="{input}")  # latest user input
    ])
    # Create the LLMChain
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    response = await chain.ainvoke({"input": userMessage})
    print(f"Response from LLM: {response}")
    logging.info(f"Response from LLM: {response}")
    history = memory.load_memory_variables({})
    print(f"Memory history after response: {history}")
    logging.info(f"Memory history after response: {history}")
    await memory.save_context({"input": userMessage}, {"response": response['text']})
    return {"response": response['text']}