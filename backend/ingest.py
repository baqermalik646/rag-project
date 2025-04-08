import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load .env variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


CSV_PATH = "data/home_depot_data_1_2021_12.csv"
OUTPUT_PATH = "data/embeddings.json"

def embed_text(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def main():
    df = pd.read_csv(CSV_PATH)
    df = df[["product_id", "title", "description"]].dropna()

    embedded_products = []

    for _, row in df.iterrows():
        product_id = row["product_id"]
        title = row["title"]
        description = row["description"]
        full_text = f"{title}. {description}"

        try:
            vector = embed_text(full_text)
            embedded_products.append({
                "product_id": str(product_id),
                "title": title,
                "description": description,
                "embedding": vector
            })
        except Exception as e:
            print(f"Failed to embed product {product_id}: {e}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(embedded_products, f)

if __name__ == "__main__":
    main()
