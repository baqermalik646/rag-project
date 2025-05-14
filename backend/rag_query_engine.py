import os
import subprocess
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()

index_dir = os.path.join("data", "faiss_index")
GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}

# --- Product Tracking Memory ---
memory_store = {}

class SessionMetadata:
    def __init__(self):
        self.last_product_doc = None

# --- LangChain Memory ---
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = {
            "history": ChatMessageHistory(),
            "meta": SessionMetadata()
        }
    return memory_store[session_id]["history"]

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

vectorstore = FAISS.load_local(index_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, "
               "formulate a standalone question. Do NOT answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for a product catalog. The context below is a list of JSON entries, "
               "each representing one product. Each entry includes fields such as product_id, sku, brand, title, price, "
               "and more. Answer the user's question using only this information when relevant. However, also consider "
               "the full chat history when responding — for example, if the user asks about a previously mentioned "
               "product, recalls their name, or references an earlier message.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt,
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain
)

rag_chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def ask_question(question: str, session_id: str = "default", stream: bool = True):
    question_clean = question.strip()
    question_lower = question_clean.lower()

    normalized = re.sub(r"[^\w\s]", " ", question_lower)
    greeting_match = any(greet in normalized for greet in GREETINGS)

    if greeting_match:
        greet_word = next((g for g in GREETINGS if g in normalized), "Hello")
        return {
            "answer": f"{greet_word.capitalize()}! How can I assist you today?",
            "sources": []
        }

    if re.search(r"\b(i have|i've got|i've|i want to ask|can i ask|i need help|i got)\b", normalized) and "question" in normalized:
        return {
            "answer": "Of course! Please go ahead and ask your product-related questions.",
            "sources": []
        }

    # logic for "its" reference to last product
    if any(p in question_lower for p in ["sku", "product_id", "price"]) and "its" in question_lower:
        last_doc = memory_store.get(session_id, {}).get("meta", {}).last_product_doc
        if last_doc:
            content = last_doc.page_content
            if "sku" in question_lower:
                match = re.search(r'"sku":\s*"?(.*?)"?[,}]', content)
                if match:
                    return {"answer": match.group(1), "sources": []}
            elif "product_id" in question_lower:
                match = re.search(r'"product_id":\s*"?(.*?)"?[,}]', content)
                if match:
                    return {"answer": match.group(1), "sources": []}
            elif "price" in question_lower:
                match = re.search(r'"price":\s*"?(.*?)"?[,}]', content)
                if match:
                    return {"answer": match.group(1), "sources": []}

    inputs = {"input": question}

    if stream:
        def response_generator():
            for chunk in rag_chain_with_history.stream(
                inputs,
                config={"configurable": {"session_id": session_id}}
            ):
                if "answer" in chunk:
                    yield chunk["answer"]
        return response_generator()
    else:
        response = rag_chain_with_history.invoke(
            inputs,
            config={"configurable": {"session_id": session_id}}
        )

        # Save last product doc for reference
        if "context" in response:
            for doc in response["context"]:
                memory_store[session_id]["meta"].last_product_doc = doc

        return {
            "answer": response["answer"],
            "sources": list(set(
                doc.metadata.get("source", "N/A")
                for doc in response.get("context", [])
            ))
        }
