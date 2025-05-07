

import os
import subprocess
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

index_dir = os.path.join("data", "faiss_index")

def faiss_index_exists(path: str) -> bool:
    return all([
        os.path.exists(os.path.join(path, "index.faiss")),
        os.path.exists(os.path.join(path, "index.pkl")),
    ])

if not faiss_index_exists(index_dir):
    print("FAISS index not found. Running ingest.py to create it...")
    subprocess.run(["python3", "ingest.py"], check=True)
else:
    print("FAISS index already exists. Skipping embedding generation.")

# Load FAISS index
vectorstore = FAISS.load_local(index_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Setting up llm
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Prompt to reformulate question using chat history
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question "
               "which might reference context in the chat history, "
               "formulate a standalone question. Do NOT answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt,
)

# QA Prompt that explicitly mentions JSON format context
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for a product catalog. "
               "The context below is a list of JSON entries, each representing one product. "
               "Each entry includes fields such as product_id, sku, brand, title, price, and more. "
               "Answer the user's question using only this information. "
               "If you cannot find an answer in the context, respond with: "
               "\"I couldn't find that in the catalog.\"\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create base RAG chain
rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain
)

memory_store = {}

def get_memory(session_id: str):
    if session_id not in memory_store:
        print(f"Creating new memory for session: {session_id}")
        memory_store[session_id] = InMemoryChatMessageHistory()
    else:
        print(f"Reusing memory for session: {session_id} — current length: {len(memory_store[session_id].messages)}")
    return memory_store[session_id]


# Wrap the chain with auto-managed memory
rag_chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# Ask question function (with session support)
import re

GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}

def ask_question(question: str, session_id: str = "default", stream: bool = True):
    question_clean = question.strip()
    question_lower = question_clean.lower()

    # Ensure session exists
    if session_id not in memory_store:
        memory_store[session_id] = SessionMemory()
    session = memory_store[session_id]

    # Normalize commas and punctuation to spaces for pattern matching
    normalized = re.sub(r"[^\w\s]", " ", question_lower)

    # Check for name anywhere
    name_match = re.search(r"\b(my name is|my nam is|i am|i'm|call me)\s+([a-zA-Z]+)\b", normalized)
    greeting_match = any(greet in normalized for greet in GREETINGS)

    if name_match:
        original_name_match = re.search(r"\b(my name is|my nam is|i am|i'm|call me)\s+([a-zA-Z]+)\b", question_clean, re.IGNORECASE)
        if original_name_match:
            name = original_name_match.group(2).capitalize()
            session.user_name = name
            greet_word = next((g for g in GREETINGS if g in normalized), "Hello") if greeting_match else "Hello"
            response_text = f"{greet_word.capitalize()}, {name}! How can I assist you today?"
            if stream:
                yield response_text
            else:
                return {
                    "answer": response_text,
                    "sources": []
                }
            return

    # 2. Greeting with remembered name
    if greeting_match:
        greet_word = next((g for g in GREETINGS if g in normalized), "Hello")
        if session.user_name:
            response_text = f"{greet_word.capitalize()}, {session.user_name}! How can I assist you today?"
        else:
            response_text = f"{greet_word.capitalize()}! How can I assist you today?"
        if stream:
            yield response_text
        else:
            return {
                "answer": response_text,
                "sources": []
            }
        return

    # 3. Handle vague intent
    if re.search(r"\b(i have|i've got|i've|i want to ask|can i ask|i need help|i got)\b", normalized) and "question" in normalized:
        response_text = "Of course! Please go ahead and ask your product-related questions."
        if stream:
            yield response_text
        else:
            return {
                "answer": response_text,
                "sources": []
            }
        return

    # 4. Default: Pass to RAG
    inputs = {"input": question}

    if stream:
        for chunk in rag_chain_with_history.stream(inputs, config={"configurable": {"session_id": session_id}}):
            if "answer" in chunk:
                yield chunk["answer"]
    else:
        response = rag_chain_with_history.invoke(inputs, config={"configurable": {"session_id": session_id}})
        return {
            "answer": response["answer"],
            "sources": list(set(doc.metadata.get("source", "N/A") for doc in response.get("context", [])))
        }
