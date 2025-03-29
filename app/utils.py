


from langchain_core.messages import HumanMessage

from langchain_core.messages import AIMessage

from langchain_core.messages import SystemMessage
from fastapi import Request, HTTPException

def format_message(msg):
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    elif isinstance(msg, AIMessage):
        return {"role": "assistant", "content": msg.content}
    elif isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    else:
        return {"role": "unknown", "content": msg.content}



def get_user_id_from_request(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")
    return user_id
