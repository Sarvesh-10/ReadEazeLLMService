import httpx
import asyncio
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "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"
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
async def test_groq():
    async with httpx.AsyncClient() as client:
        payload = {
        "model": "llama-3.2-3b-preview",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "stream": True
    }
        async with client.stream("POST", GROQ_API_URL, headers=HEADERS, json=payload) as response:
            async for line in response.aiter_lines():
                print(line)  # ✅ Print each line as received

asyncio.run(test_groq())