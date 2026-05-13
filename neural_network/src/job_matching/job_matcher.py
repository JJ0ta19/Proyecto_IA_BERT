import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from ..ml_models.embedding_model import EmbeddingModel

class JobMatcher:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.job_embeddings = None
        self.jobs_df = None

    def fit(self, jobs_df: pd.DataFrame, text_column: str = "job_description"):
        print("Fitting job matcher...")
        self.jobs_df = jobs_df.copy()
        texts = jobs_df[text_column].fillna("").astype(str).tolist()
        self.job_embeddings = self.embedding_model.encode(texts)
        print(f"Fitted with {len(texts)} jobs")

    def match(self, resume_text: str, top_k: int = 5) -> List[Dict]:
        if self.job_embeddings is None:
            raise ValueError("Job matcher not fitted. Call fit() first.")

        resume_embedding = self.embedding_model.encode_single(resume_text)

        similarities = []
        for idx, job_emb in enumerate(self.job_embeddings):
            sim = self._cosine_similarity(resume_embedding, job_emb)
            similarities.append((idx, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in similarities[:top_k]:
            job_info = self.jobs_df.iloc[idx].to_dict()
            results.append({
                'job_index': idx,
                'compatibility_score': round(score * 100, 2),
                'job_info': job_info
            })

        return results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0

    def compute_skill_match(self, resume_skills: List[str], job_skills: List[str]) -> Dict:
        resume_skills_lower = [s.lower() for s in resume_skills]
        job_skills_lower = [s.lower() for s in job_skills]

        matched = set(resume_skills_lower) & set(job_skills_lower)
        missing = set(job_skills_lower) - set(resume_skills_lower)

        match_percentage = len(matched) / len(job_skills_lower) * 100 if job_skills_lower else 0

        return {
            'matched_skills': list(matched),
            'missing_skills': list(missing),
            'match_percentage': round(match_percentage, 2)
        }

    def save(self, path: str):
        np.save(path, self.job_embeddings)

    def load(self, path: str, jobs_df: pd.DataFrame):
        self.job_embeddings = np.load(path)
        self.jobs_df = jobs_df