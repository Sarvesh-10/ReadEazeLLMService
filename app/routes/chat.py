from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from ..services.memory import ChatMemory, get_chat_memory
from ..services.llm import streamLLMResponses
from ..auth import get_current_user
from ..utils import format_message, get_user_id_from_request
from fastapi import APIRouter, HTTPException
from ..customLogging import logger



router = APIRouter(prefix='/chat',tags=['chat'])

@router.post('/{book_id}')
async def chat_with_book(book_id:str,request:Request):
    logger.info(f"Received chat request for book_id: {book_id}", extra={"request": request.headers})
    user_id = getattr(request.state, 'user_id', None)
    logger.info(f"User ID from request: {user_id}")
    logger.info(f"Book ID from request: {book_id}")
    data = await request.json()
    logger.info(f"Request data: {data}")
    user_message = data.get("userMessage")
    logger.info(f"User message: {user_message}")
    if not user_message:
        logger.error("No user message provided in the request")
        return {"error": "No message provided"}
    system_message = data.get("systemMessage")
    logger.info(f"System message: {system_message}")
    return await streamLLMResponses(user_id=user_id,book_id=book_id,systemMessage=system_message,userMessage=user_message)
@router.options('/{book_id}')
def prefligh_handler(book_id:str,req:Request):
    print("Got book id",book_id)
    return {"message":"Preflight Ok"}


@router.get('/get-history/book/{book_id}')
async def get_chat_history(book_id:str,request:Request):    
    try:
        user_id = get_user_id_from_request(request)  # ✅ Clean one-liner!
        memory = get_chat_memory(user_id, book_id)
        messages = [format_message(msg) for msg in memory.get_messages()]
        return {"status": "success", "history": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chat history: {e}")
    

