import os
import numpy as np
from mistralai.client import Mistral

EMBED_MODEL = "mistral-embed"
BATCH_SIZE = 32


class MistralEmbedder:
    def __init__(self):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        all_vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = self.client.embeddings.create(model=EMBED_MODEL, inputs=batch)
            all_vectors.extend(item.embedding for item in response.data)
        return np.array(all_vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
