import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name).to(self.device)
        self.model_name = model_name

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def compute_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)
        return self._cosine_similarity(emb1, emb2)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)

    def find_similar(self, query: str, corpus: List[str], top_k: int = 5) -> List[tuple]:
        query_embedding = self.encode_single(query)
        corpus_embeddings = self.encode(corpus, show_progress=False)

        similarities = []
        for idx, emb in enumerate(corpus_embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            similarities.append((idx, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def save_embeddings(self, embeddings: np.ndarray, path: str):
        np.save(path, embeddings)

    def load_embeddings(self, path: str) -> np.ndarray:
        return np.load(path)