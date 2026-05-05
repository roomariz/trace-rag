import requests
from pymilvus import MilvusClient

OLLAMA_BASE_URL = "http://localhost:11435"
EMBED_MODEL = "mxbai-embed-large"

# ---- Ollama embedding function (with debug + timeout) ----

def embed(texts):
    vectors = []

    for text in texts:
        print(f"[Embedding] {text}", flush=True)

        res = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": EMBED_MODEL,
                "input": [text]
            },
            timeout=30
        )

        res.raise_for_status()
        data = res.json()

        if "embeddings" in data and len(data["embeddings"]) > 0:
            embedding = data["embeddings"][0]
        else:
            raise RuntimeError(f"Embedding failed: {data}")

        print(f"[Done] dim={len(embedding)}", flush=True)

        vectors.append(embedding)

    return vectors

# ---- Start ----

print("Script started...", flush=True)


# ---- Connect to Milvus ----

# client = MilvusClient(uri="http://localhost:19530")
print("Connecting to Milvus...", flush=True)

client = MilvusClient(uri="tcp://localhost:19530")

print("Connected to Milvus", flush=True)

# ---- Reset collection ----

if client.has_collection(collection_name="demo_collection"):
    print("Dropping existing collection...", flush=True)
    client.drop_collection(collection_name="demo_collection")


# ---- Sample data ----

docs = [
    "Python is widely used for data science and machine learning.",
    "Neural networks are inspired by the human brain structure.",
    "Databases store structured and unstructured information efficiently.",
]


# ---- Generate embeddings ----

vectors = embed(docs)

dim = len(vectors[0])
print(f"Vector dimension: {dim}", flush=True)


# ---- Create collection dynamically ----

client.create_collection(
    collection_name="demo_collection",
    dimension=dim,
    metric_type="COSINE"
)

print("Collection created", flush=True)


# ---- Insert data ----

next_id = 0

data = [
    {
        "id": next_id + i,
        "vector": vectors[i],
        "text": docs[i],
        "subject": "tech"
    }
    for i in range(len(vectors))
]

next_id += len(data)

res = client.insert(collection_name="demo_collection", data=data)

client.load_collection(collection_name="demo_collection")  # ← ADD THIS

print("Insert result:", res, flush=True)


# ---- Query ----

query = ["What is machine learning?"]
query_vectors = embed(query)

res = client.search(
    collection_name="demo_collection",
    data=query_vectors,
    limit=2,
    output_fields=["text", "subject"],
)

print("\nSearch results:", flush=True)
for r in res[0]:
    print(r, flush=True)


# ---- Add more data ----

docs_bio = [
    "The human heart pumps blood throughout the body.",
    "Photosynthesis allows plants to convert sunlight into energy.",
    "DNA carries genetic information in living organisms.",
]

vectors_bio = embed(docs_bio)

data_bio = [
    {
        "id": next_id + i,
        "vector": vectors_bio[i],
        "text": docs_bio[i],
        "subject": "biology"
    }
    for i in range(len(vectors_bio))
]

next_id += len(data_bio)

client.insert(collection_name="demo_collection", data=data_bio)

client.flush(collection_name="demo_collection")
client.load_collection(collection_name="demo_collection")

print("Inserted biology data", flush=True)


# ---- Filtered search ----

query_vector = embed(["Explain biological processes"])

res = client.search(
    collection_name="demo_collection",
    data=query_vector,
    filter="subject == 'biology'",
    limit=2,
    output_fields=["text", "subject"],
)

print("\nFiltered search results:", flush=True)
for r in res[0]:
    print(r, flush=True)


# ---- Add domain-specific biology documents ----

# ---- Add domain-specific biology documents ----

bio_docs = [
    "Machine learning has been used for drug design.",
    "Computational synthesis with AI algorithms predicts molecular properties.",
    "DDR1 is involved in cancers and fibrosis.",
]

bio_vectors = embed(bio_docs)

start_id = 100  # avoids collision

bio_data = [
    {
        "id": start_id + i,
        "vector": bio_vectors[i],
        "text": bio_docs[i],
        "subject": "biology"
    }
    for i in range(len(bio_docs))
]

client.insert(collection_name="demo_collection", data=bio_data)

client.flush(collection_name="demo_collection")
client.load_collection(collection_name="demo_collection")

print("Inserted advanced biology documents", flush=True)


# ---- Filtered semantic search ----

query_text = "Tell me about AI in biology"
query_vector = embed([query_text])

search_results = client.search(
    collection_name="demo_collection",
    data=query_vector,
    filter="subject == 'biology'",
    limit=3,
    output_fields=["text", "subject"],
)

print("\nBiology-focused search results:", flush=True)

for hit in search_results[0]:
    print({
        "score": hit["distance"],
        "text": hit["entity"]["text"],
        "subject": hit["entity"]["subject"]
    }, flush=True)

"""
Next step (where real value begins)

Right now you only retrieve documents.

Next level is:

1. Retrieval + generation (RAG)
   take top-k results
   send to LLM
   generate answer

2. Evaluation
   are results actually relevant?
   measure semantic accuracy

3. Optimisation
   indexing (IVF, HNSW)
   vector normalisation
   reranking

Bottom line

You've successfully:
- debugged infra (Docker, Milvus, Ollama)
- fixed API mismatches
- built a working semantic search system

That is not trivial. This is exactly how real systems are built.
"""