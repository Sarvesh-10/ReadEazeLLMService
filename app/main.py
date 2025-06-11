from fastapi import FastAPI,Request
from fastapi.responses import StreamingResponse
from .services.llm import streamLLMResponses
from fastapi.middleware.cors import CORSMiddleware
from .middleware import JWTAuthMiddleware
from .routes.chat import router as chat_router
from .routes.image import router as image_router
app = FastAPI()

app.add_middleware(JWTAuthMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(image_router)
app.include_router(chat_router)


@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/test-cookie")
def test_cookie(request: Request):
    token = request.cookies.get('token')
    print(f"Received token: {token}")  # ✅ Check if token is received
    return {"token": token}