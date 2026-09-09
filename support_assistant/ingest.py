import os
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
DB_DIR = "chroma_db"
COLLECTION_NAME = "zepto_policies"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

documents = []
ids = []
metadatas = []

for filename in sorted(os.listdir(DOCS_DIR)):
    if filename.endswith(".txt"):
        filepath = os.path.join(DOCS_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read().strip()

        documents.append(text)
        ids.append(filename.replace(".txt", ""))
        metadatas.append({
            "document_id": filename.replace(".txt", ""),
            "source": filename
        })

embeddings = model.encode(
    documents,
    normalize_embeddings=True
).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Documents indexed:", collection.count())

print("\nIndexed documents:")

for item in collection.get()["ids"]:
    print(item)

print("\nChromaDB collection:", COLLECTION_NAME)
print("Database directory:", DB_DIR)