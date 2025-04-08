import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# Path to saved embeddings
EMBEDDING_FILE = "data/embeddings.json"

def load_embeddings():
    with open(EMBEDDING_FILE, "r") as f:
        return json.load(f)

def embed_query(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_product(query, top_k=1):
    query_embedding = embed_query(query)
    products = load_embeddings()

    scored = []
    for product in products:
        score = cosine_similarity(query_embedding, product["embedding"])
        scored.append((score, product))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [item[1] for item in scored[:top_k]]

