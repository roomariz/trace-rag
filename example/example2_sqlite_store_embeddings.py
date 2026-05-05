import numpy as np
import sqlite3

# Import from your existing file
from example1_ollama_vectorize import TextVectorizer

# ---- Save embeddings ----

def save_embeddings_to_db(embeddings, db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            embedding BLOB
        )
    ''')

    for text, embedding in embeddings.items():
        if embedding is None:
            continue

        embedding_bytes = embedding.astype("float32").tobytes()
        cursor.execute(
            "INSERT INTO embeddings (text, embedding) VALUES (?, ?)",
            (text, embedding_bytes)
        )

    conn.commit()
    conn.close()


# ---- Read embeddings ----

def read_embeddings_from_db(db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT text, embedding FROM embeddings")
    rows = cursor.fetchall()

    embeddings = {}
    for row in rows:
        text = row[0]
        embedding_bytes = row[1]

        # Convert bytes back to NumPy array
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

        embeddings[text] = embedding

    conn.close()
    return embeddings


# ---- Usage ----

vectorizer = TextVectorizer()

texts = [
    "The food was delicious and the waiter...",
    "The movie was amazing, the acting was superb.",
    "I had a terrible experience at the hotel."
]

embeddings_dict = {}

for text in texts:
    embeddings_dict[text] = vectorizer.vectorize(text)

save_embeddings_to_db(embeddings_dict)

print("Embeddings saved to SQLite successfully.")


# ---- Read back from DB ----

retrieved_embeddings = read_embeddings_from_db()

print(retrieved_embeddings)

for text, embedding in retrieved_embeddings.items():
    print(f"Text: {text}")
    print(f"Embedding (first 5): {embedding[:5]}")
    print(f"Length: {len(embedding)}")
    print("-----")


# ---- Final validation ----

from sklearn.metrics.pairwise import cosine_similarity

values = list(retrieved_embeddings.values())

v1 = values[0]
v2 = values[1]

print("Cosine similarity:")
print(cosine_similarity([v1], [v2]))