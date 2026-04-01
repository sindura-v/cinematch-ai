import pandas as pd
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="watchlist")

df = pd.read_csv("watchlist.csv", skipinitialspace=True)
print(f"Loaded {len(df)} entries from watchlist")
print(df.head())

def create_text(row):
    return f"{row['title']} is a {row['type']} in the {row['genre']} genre. Mood: {row['mood']}. Rating: {row['rating']}/10. {row['notes']}"

print("Creating embeddings and storing in ChromaDB...")

for index, row in df.iterrows():
    # 1. Create the text description
    text = create_text(row)
    
    # 2. Get embedding from OpenAI
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    
    # 3. Store in ChromaDB
    collection.add(
        ids=[str(index)],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "title": row["title"],
            "type": row["type"],
            "genre": row["genre"],
            "platform": row["platform"],
            "rating": float(row["rating"]),
            "mood": row["mood"]
        }]
    )
    print(f"  Stored: {row['title']}")

print(f"\nDone! {len(df)} entries stored in ChromaDB.")