
# Customer Chatbot

## Project Overview

This is a Retrieval-Augmented Generation (RAG)-based chatbot designed to answer customer questions about retail products using OpenAI’s GPT model. The chatbot integrates semantic search over real product data (e.g., prices, SKUs, brand names) and delivers context-aware, multi-turn conversations using FastAPI for the backend and Next.js for the frontend.

---

## Features

- **RAG with Semantic Search:** Uses FAISS and OpenAI embeddings to retrieve relevant product entries for customer queries.
- **Context-Aware Conversations:** Maintains chat history across turns using `RunnableWithMessageHistory` to support natural dialogue (e.g., "What’s its price?" after a prior product mention).
- **Follow-Up Precision:** Remembers the last discussed product to support structured follow-up questions (e.g., SKU, product ID, brand, availability).
- **Adaptive LLM Responses:** Handles user preferences like concise answers, greetings, and name recognition.
- **FastAPI Backend:** Provides a `/chat` endpoint with session-aware interactions.
- **Streaming Capable:** Backend supports token streaming for real-time responses (optional).
- **Frontend with Next.js:** Responsive UI for customer chat experience.

---

## How to Run

### Backend (FastAPI)

1. **Navigate to the backend folder:**
   ```bash
   cd backend
   
2. Create and activate a virtual environment:

   python3 -m venv venv
source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Add your OpenAI API key to a .env file:
   OPENAI_API_KEY=your_openai_key

5. Generate vector embeddings from product data (may take a while):
   python3 ingest.py

6. Start the backend server:
   python -m uvicorn main:app --reload

### Frontend (Next.js)
1. Navigate to the frontend folder:
   cd frontend/my-app
2. Install frontend dependencies:
   npm install
3. Run the frontend server:
   npm run dev

