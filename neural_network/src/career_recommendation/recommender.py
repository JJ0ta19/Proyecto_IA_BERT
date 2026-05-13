import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from ..ml_models.embedding_model import EmbeddingModel

class JobRecommender:
    def __init__(self, embedding_model: EmbeddingModel, occupations_df: Optional[pd.DataFrame] = None, skill_relations_df: Optional[pd.DataFrame] = None):
        self.embedding_model = embedding_model
        self.occupations_df = occupations_df
        self.skill_relations_df = skill_relations_df
        self.occupation_embeddings = None

    def fit_occupations(self):
        if self.occupations_df is None:
            print("No occupations data available")
            return

        print("Fitting occupation recommender...")
        texts = self.occupations_df.iloc[:, 0].fillna("").astype(str).tolist()
        self.occupation_embeddings = self.embedding_model.encode(texts)

    def recommend_jobs(self, predicted_category: str, resume_skills: List[str], top_k: int = 5) -> List[Dict]:
        if self.occupation_embeddings is None:
            return []

        category_embedding = self.embedding_model.encode_single(predicted_category)

        similarities = []
        for idx, occ_emb in enumerate(self.occupation_embeddings):
            sim = self._cosine_similarity(category_embedding, occ_emb)
            similarities.append((idx, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        for idx, score in similarities[:top_k]:
            occ_name = self.occupations_df.iloc[idx].to_dict()
            recommendations.append({
                'occupation': occ_name,
                'similarity_score': round(score * 100, 2)
            })

        return recommendations

    def suggest_skill_gaps(self, resume_skills: List[str], target_occupation: str) -> List[str]:
        if self.skill_relations_df is None:
            return []

        resume_skills_lower = [s.lower() for s in resume_skills]

        matching_rows = self.skill_relations_df[
            self.skill_relations_df.iloc[:, 0].str.lower().str.contains(target_occupation.lower(), na=False)
        ]

        if matching_rows.empty:
            return []

        required_skills = matching_rows.iloc[:, 1].str.lower().tolist() if len(matching_rows.columns) > 1 else []

        gap_skills = [skill for skill in required_skills if skill.lower() not in resume_skills_lower]
        return gap_skills[:10]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0

    def get_career_paths(self, current_category: str) -> List[Dict]:
        if self.occupation_embeddings is None:
            return []

        current_embedding = self.embedding_model.encode_single(current_category)

        similarities = []
        for idx, occ_emb in enumerate(self.occupation_embeddings):
            sim = self._cosine_similarity(current_embedding, occ_emb)
            similarities.append((idx, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [{"index": idx, "similarity": round(score, 2)} for idx, score in similarities[1:6]]