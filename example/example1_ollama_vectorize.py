import warnings
import requests

warnings.filterwarnings("ignore")

OLLAMA_BASE_URL = "http://localhost:11434"

import numpy as np
import sqlite3

class TextVectorizer:
    def __init__(self, base_url="http://localhost:11434"):
        """
        Initialize the vectorizer with Ollama local endpoint.

        :param base_url: Ollama server URL
        """
        self.base_url = base_url

        # Embedding model available in your setup
        self.model = "nomic-embed-text"

    def vectorize(self, text):
        """
        Convert input text into a vector using Ollama embeddings.

        :param text: Input text to vectorize
        :return: Numpy array of the text embedding
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )

            response.raise_for_status()
            embedding = response.json()["embedding"]

            return np.array(embedding)

        except Exception as e:
            print(f"Error during vectorization: {e}")
            return None

    def compare_vectors(self, vector1, vector2):
        """
        Calculate cosine similarity between two vectors.

        :param vector1: First vector
        :param vector2: Second vector
        :return: Cosine similarity score
        """
        if vector1 is None or vector2 is None:
            return None

        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)

        return dot_product / (norm_vector1 * norm_vector2)
    
text_vectorizer = TextVectorizer()

embedding = text_vectorizer.vectorize("The food was delicious and the waiter...")

print(embedding[:10])

print(len(embedding))