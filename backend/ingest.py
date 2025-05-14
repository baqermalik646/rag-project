import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm
import json
import os

# Load environment
load_dotenv()
csv_path = "data/home_depot_data_1_2021_12.csv"
df = pd.read_csv(csv_path)

documents = []
for _, row in df.iterrows():
    row_dict = row.dropna().to_dict()
    json_str = json.dumps(row_dict, ensure_ascii=False)
    documents.append(Document(page_content=json_str, metadata={"source": csv_path}))

embedding_model = OpenAIEmbeddings()

def batch_faiss_index(docs, batch_size=100):
    sub_vectorstores = []
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding"):
        batch = docs[i:i + batch_size]
        sub_vs = FAISS.from_documents(batch, embedding_model)
        sub_vectorstores.append(sub_vs)

    merged_vectorstore = sub_vectorstores[0]
    for sub_vs in sub_vectorstores[1:]:
        merged_vectorstore.merge_from(sub_vs)

    return merged_vectorstore

vectorstore = batch_faiss_index(documents)
save_path = "data/faiss_index"
vectorstore.save_local(folder_path=save_path)
print(f"FAISS index saved to '{save_path}'")
