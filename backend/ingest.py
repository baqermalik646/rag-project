
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import json

# Loading environment
load_dotenv()

csv_path = "data/home_depot_data_1_2021_12.csv"

# Loading the dataset
df = pd.read_csv(csv_path)

# Convert each row into a JSON string (preserving all columns)
documents = []
for _, row in df.iterrows():
    row_dict = row.dropna().to_dict()  # dropping NaNs to reduce noise
    json_str = json.dumps(row_dict, ensure_ascii=False)
    documents.append(Document(page_content=json_str, metadata={"source": csv_path}))

print(f"Loaded and converted {len(documents)} product records into JSON documents.")

# Creating embeddings
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embedding_model)

# Save FAISS index
save_path = "data/faiss_index"
vectorstore.save_local(folder_path=save_path)
print(f"FAISS index saved to '{save_path}'")
