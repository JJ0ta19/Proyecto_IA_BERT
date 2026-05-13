"""
Sistema Inteligente de Análisis de Hojas de Vida
Pipeline completo: NLP + Transformers + Matching + Recomendación
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.data_loader import DatasetLoader
from src.data_processing.preprocessor import DatasetPreprocessor
from src.ml_models.bert_classifier import BertClassifierModel
from src.ml_models.embedding_model import EmbeddingModel
from src.job_matching.job_matcher import JobMatcher
from src.career_recommendation.recommender import JobRecommender
from src.text_preprocessing.text_cleaner import TextCleaner
from src.text_preprocessing.skill_extractor import SkillExtractor
from src.config_utils.config import OUTPUT_DIR, CHECKPOINT_DIR

class ResumeAnalysisSystem:
    def __init__(self, use_gpu: bool = True):
        self.device = "cuda" if use_gpu and os.name != "nt" else "cpu"
        print(f"Initializing system on device: {self.device}")

        self.text_cleaner = TextCleaner()
        self.skill_extractor = None
        self.classifier = None
        self.embedding_model = None
        self.job_matcher = None
        self.recommender = None
        self.preprocessor = None
        self.label_mapping = None

    def initialize(self):
        print("\n=== Loading and Preparing Data ===")
        loader = DatasetLoader()
        datasets = loader.load_all()

        self.preprocessor = DatasetPreprocessor()
        skills_list = loader.get_skills_list()
        self.skill_extractor = SkillExtractor(skills_list)
        
        # Cargar mapeo Job Role -> Category
        self.job_role_category_map = loader.get_job_role_category_map()
        print(f"  - Loaded {len(self.job_role_category_map)} Job Role -> Category mappings")

        print("\n=== Initializing Embedding Model ===")
        self.embedding_model = EmbeddingModel(device=self.device)

        print("\n=== System Ready ===")

    def train_classifier(self, epochs: int = 3, batch_size: int = 16):
        print("\n=== Training Classifier ===")
        loader = DatasetLoader()
        datasets = loader.load_all()

        if "resume_class_training" in datasets:
            df = datasets["resume_class_training"]
            text_col = "Resume Text"
            label_col = "Job Role"
        else:
            print("No training data found!")
            return

        texts, labels = self.preprocessor.prepare_classification_data(df, text_col, label_col)
        train_texts, val_texts, train_labels, val_labels = self.preprocessor.prepare_for_bert(texts, labels)

        self.label_mapping, self.reverse_mapping = self.preprocessor.get_label_mapping()
        num_classes = len(self.label_mapping)

        print(f"\nTraining with {num_classes} classes...")
        self.classifier = BertClassifierModel(num_classes=num_classes, device=self.device)
        self.classifier.train(
            train_texts, train_labels,
            val_texts, val_labels,
            epochs=epochs,
            batch_size=batch_size,
            save_path=os.path.join(CHECKPOINT_DIR, "bert_classifier.pt")
        )

    def fit_matcher(self, jobs_df):
        print("\n=== Fitting Job Matcher ===")
        self.job_matcher = JobMatcher(self.embedding_model)
        self.job_matcher.fit(jobs_df, text_column="job_description" if "job_description" in jobs_df.columns else jobs_df.columns[0])

    def fit_recommender(self):
        print("\n=== Fitting Recommender ===")
        loader = DatasetLoader()
        occupations_df = loader.get_occupations()
        skill_relations_df = loader.get_skill_relations()

        self.recommender = JobRecommender(self.embedding_model, occupations_df, skill_relations_df)
        self.recommender.fit_occupations()

    def analyze_resume(self, resume_text: str, num_recommendations: int = 5):
        print("\n=== Analyzing Resume ===")

        cleaned_text = self.text_cleaner.clean(resume_text)
        print(f"1. Text cleaned: {len(cleaned_text)} characters")

        # Extraer Education y Experience Years con regex
        education = self._extract_education(resume_text)
        experience_years = self._extract_experience_years(resume_text)
        print(f"2. Extracted Education: {education}")
        print(f"   Extracted Experience: {experience_years} años")

        # Predicción de Job Role
        predicted_job_role = None
        predicted_category = None
        if self.classifier:
            prediction = self.classifier.predict([cleaned_text])[0]
            predicted_job_role = self.reverse_mapping.get(prediction, "Unknown")
            predicted_category = self._get_category_from_job_role(predicted_job_role)
            print(f"3. Predicted Job Role: {predicted_job_role}")
            print(f"   Derived Category: {predicted_category}")

        # Extraer Skills
        skills = []
        if self.skill_extractor:
            skills = self.skill_extractor.extract_from_resume(resume_text)
            print(f"4. Extracted Skills ({len(skills)}): {skills[:10]}...")

        # Job Matches
        matches = []
        if self.job_matcher:
            matches = self.job_matcher.match(cleaned_text, top_k=num_recommendations)
            print(f"5. Top {num_recommendations} Job Matches:")
            for match in matches:
                print(f"   - Score: {match['compatibility_score']}%")

        # Career Recommendations
        recommendations = []
        if self.recommender and predicted_category:
            recommendations = self.recommender.recommend_jobs(predicted_category, skills if self.skill_extractor else [], top_k=num_recommendations)
            print(f"6. Career Recommendations:")
            for rec in recommendations:
                print(f"   - {rec}")

        return {
            "cleaned_text": cleaned_text,
            "education": education,
            "experience_years": experience_years,
            "predicted_job_role": predicted_job_role,
            "predicted_category": predicted_category,
            "extracted_skills": skills,
            "job_matches": matches,
            "recommendations": recommendations
        }

    def _extract_education(self, text: str) -> str:
        import re
        text_lower = text.lower()
        education_patterns = [
            r"bachelor'?s?\s+in\s+([a-z\s]+)",
            r"master'?s?\s+in\s+([a-z\s]+)",
            r"phd\s+in\s+([a-z\s]+)",
            r"doctorate\s+in\s+([a-z\s]+)",
            r"associate\s+degree\s+in\s+([a-z\s]+)",
            r"high\s+school\s+diploma",
            r"mba",
            r"bsc|bs|msc|ms",
        ]
        for pattern in education_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(0).upper()
        return "Not specified"

    def _extract_experience_years(self, text: str) -> int:
        import re
        text_lower = text.lower()
        patterns = [
            r"(\d+)\s+years?\s+of\s+experience",
            r"(\d+)\s+year\s+experience",
            r"experience:\s*(\d+)\s+years?",
            r"(\d+)\+?\s+years",
        ]
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        return 0

    def _get_category_from_job_role(self, job_role: str) -> str:
        if hasattr(self, 'job_role_category_map') and self.job_role_category_map:
            category = self.job_role_category_map.get(job_role)
            if category:
                return category
        return "Unknown"

def main():
    system = ResumeAnalysisSystem(use_gpu=False)
    system.initialize()

    sample_resume = """
    EXPERIENCE
    Software Engineer at Tech Corp
    - Developed Python applications using Django and Flask
    - Worked with PostgreSQL, MongoDB, and Redis
    - Deployed applications using AWS and Docker
    - Collaborated with team using Git and CI/CD pipelines

    SKILLS
    Python, JavaScript, Django, Flask, React, AWS, Docker, PostgreSQL, Git

    EDUCATION
    BS Computer Science - University of Technology
    """

    result = system.analyze_resume(sample_resume)
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()