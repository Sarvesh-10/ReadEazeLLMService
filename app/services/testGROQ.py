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

system_prompt = """You are a memory compression assistant. Your job is to maintain a running summary of a conversation.

- You will be given a previous summary (if any).
- You will also be given the next few turns of conversation.
- Your task is to return an updated summary that integrates the new information with the previous summary.
- Be concise and preserve key context."""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Previous summary:\nNone"},
    {"role": "user", "content": "Here are the next few turns of the conversation:"},
    {"role": "user", "content": "Hi I am Akhil"},
    {"role": "assistant", "content": "Nice to meet you, Akhil! 😊 I'm your friendly AI sidekick..."},
    {"role": "user", "content": "do you recollect my name?"},
    {"role": "assistant", "content": "I do! Your name is Akhil..."},
    {"role": "user", "content": "i am reading the daily stoic book now"},
    {"role": "assistant", "content": "The Daily Stoic is an amazing book! Ryan Holiday’s writing..."}
]

async def call_groq():
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 300,
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