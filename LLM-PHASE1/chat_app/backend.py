import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = "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"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    
    try:
        messages = req.history + [{"role": "user", "content": req.message}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        reply = response.choices[0].message["content"]
        print("Reply:", reply)
        return {"reply": reply}
    except Exception as e:
        print("Error:", e)
        return {"error": str(e)} 