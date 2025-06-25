from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from ..services.memory import ChatMemory, get_chat_memory
from ..services.stream_response import streamLLMResponses
from ..auth import get_current_user
from ..utils import format_message, get_user_id_from_request
from fastapi import APIRouter, HTTPException



router = APIRouter(prefix='/chat',tags=['chat'])

@router.post('/{book_id}')
async def chat_with_book(book_id:str,request:Request):
    print("HERE IN CHAT WITH BOOK")
    user_id = getattr(request.state, 'user_id', None)
    print(user_id,book_id)
    data = await request.json()
    user_message = data.get("userMessage")
    if not user_message:
        return {"error": "No message provided"}
    system_message = data.get("systemMessage")
    return streamLLMResponses(user_id=user_id,book_id=book_id,systemMessage=system_message,userMessage=user_message)
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
    

