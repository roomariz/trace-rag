import requests
import numpy as np

OLLAMA_BASE_URL = "http://localhost:11434"

def get_embedding(text, model="nomic-embed-text"):
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={
                "model": model,
                "prompt": text
            }
        )

        response.raise_for_status()
        embedding = response.json()["embedding"]

        return np.array(embedding)

    except Exception as e:
        print(f"Error: {e}")
        return None


# ---- Example usage ----

text = "The food was delicious and the waiter was very polite."

embedding = get_embedding(text)

print("First 10 values:")
print(embedding[:10])

print("Length:")
print(len(embedding))