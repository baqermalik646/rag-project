from fastapi import FastAPI
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os

from rag_query_engine import retrieve_relevant_product

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    product_title: str
    product_description: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_question = request.message

    top_product = retrieve_relevant_product(user_question)[0]

    context = f"""
    You are a helpful assistant for customers.
    The customer asked: "{user_question}"

    Here's a related product you can use to help answer:
    Title: {top_product['title']}
    Description: {top_product['description']}
    """

    response = client.chat.completions.create(
        #model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Home Depot customers."},
            {"role": "user", "content": context}
        ]
    )

    answer = response.choices[0].message.content.strip()

    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
        product_title=top_product["title"],
        product_description=top_product["description"]
    )

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            session_id = data.get("session_id")
            message = data.get("message")

            top_product = retrieve_relevant_product(message)[0]

            context = f"""
            The customer asked: "{message}"
            Use ONLY the following product to answer:
            Title: {top_product['title']}
            Description: {top_product['description']}
            """

            response = client.chat.completions.create(
                #model="gpt-3.5-turbo",
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for customers."},
                    {"role": "user", "content": context}
                ],
                stream=True
            )

            for chunk in response:
                answer = chunk.choices[0].delta.content or ""
                await websocket.send_json({
                    "session_id": session_id,
                    "answer": answer,
                    "product_title": top_product["title"],
                    "product_description": top_product["description"]
                })

    except WebSocketDisconnect:
        print("WebSocket disconnected")
