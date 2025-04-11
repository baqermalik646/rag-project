# rag_query_engine.py

import os
import subprocess
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# Check if FAISS index exists, else run ingest.py
index_dir = os.path.join("data", "faiss_index")
index_file = os.path.join(index_dir, "faiss.index")

if not os.path.exists(index_file):
    print("FAISS index not found. Running ingest.py to create it...")
    subprocess.run(["python3", "ingest.py"], check=True)

# Load FAISS index
vectorstore = FAISS.load_local(index_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Setting up llm and streaming enabled
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4, streaming=True)

# Token-limited summarization memory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=1000
)

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

# Creating final RAG chain
rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain
)


def ask_question(question: str, stream: bool = True):
    inputs = {
        "input": question,
        "chat_history": memory.chat_memory.messages
    }

    if stream:
        final_answer = ""
        for chunk in rag_chain.stream(inputs):
            if "answer" in chunk:
                final_answer += chunk["answer"]
                yield chunk["answer"]
        memory.save_context({"input": question}, {"answer": final_answer})
    else:
        response = rag_chain.invoke(inputs)
        memory.save_context({"input": question}, {"answer": response["answer"]})
        return {
            "answer": response["answer"],
            "sources": list(set(doc.metadata.get("source", "N/A") for doc in response.get("context", [])))
        }
