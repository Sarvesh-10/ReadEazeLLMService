import httpx
import asyncio
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# from langchain_groq import ChatGroq
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = ""
HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
# summaryLLM = ChatGroq(
#     model="llama3-8b-8192",
#     api_key=GROQ_API_KEY,
#     max_tokens=4000,
#     streaming=False
# )

system_prompt = """Summarize the conversation below, combining it with any previous summary. Keep it short and focused. Only return the updated summary — no explanation.
This is a conversation and summary between a user and an AI assistant. Please summarize this in a neutral tone, without any personal opinions or biases. The summary should be concise and to the point, capturing the main ideas and key points discussed in the conversation."""
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Previous summary:\nNone"},
    {"role": "user", "content": "Here are the next few turns of the conversation:"},
    {"role": "user", "content": "Hi I am trying to read the Introduction to linear algebra and I am on dot product..."},
    {"role": "assistant", "content": "You're on the dot product section..."},
    {"role": "user", "content": "Put a weight of 4 at x = -1..."},
    {"role": "assistant", "content": "**The See-Saw Example**..."},
    {"role": "user", "content": "This example is typical of engineering and science..."},
    {"role": "assistant", "content": "**Moments and Balance**..."},
    {"role": "user", "content": "Please summarize the conversation so far.I hav also given the previous summary if it exists"}
]

async def call_groq():
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        # "temperature": 0.7,
        # "max_tokens": 300,
        "stream": False
    }

    async with httpx.AsyncClient() as client:
        res = await client.post(GROQ_API_URL, headers=HEADERS, json=payload)

    print("\n=== STATUS ===", res.status_code)
    if res.status_code != 200:
        print("ERROR:\n", res.text)
        return

    data = res.json()
    print("data:\n", data)
    output = data['choices'][0]['message']['content']
    print("\n=== GENERATED SUMMARY ===\n")
    print(output.strip())

asyncio.run(call_groq())