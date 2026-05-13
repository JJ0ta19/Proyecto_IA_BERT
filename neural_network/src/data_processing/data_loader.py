import pandas as pd
import os
from typing import Dict, Tuple, Optional
from ..config_utils.config import DATASET_PATHS, DATASET_DIR

class DatasetLoader:
    def __init__(self):
        self.datasets = {}

    def load_all(self) -> Dict[str, pd.DataFrame]:
        print("Loading datasets...")
        self._load_resume_classification()
        print(f"Loaded {len(self.datasets)} datasets")
        return self.datasets

    def _load_resume_classification(self):
        print("Loading 1_resume_classification...")
        training_path = DATASET_PATHS["1_resume_classification_training"]
        jobs_path = DATASET_PATHS["1_resume_classification_jobs"]
        skills_path = DATASET_PATHS["1_resume_classification_skills"]

        if os.path.exists(training_path):
            df = pd.read_csv(training_path)
            self.datasets["resume_class_training"] = df
            print(f"  - training_data.csv: {len(df)} rows, {len(df.columns)} columns")
            
            # Crear mapeo Job Role -> Category
            self.datasets["job_role_category_map"] = df[["Job Role", "Category"]].drop_duplicates()

        if os.path.exists(jobs_path):
            df = pd.read_csv(jobs_path)
            self.datasets["job_roles"] = df
            print(f"  - job_roles.csv: {len(df)} rows")

        if os.path.exists(skills_path):
            df = pd.read_csv(skills_path)
            self.datasets["skills_list"] = df
            print(f"  - skills_list.csv: {len(df)} rows")

    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        return self.datasets.get(name)

    def get_training_data(self) -> pd.DataFrame:
        if "resume_class_training" in self.datasets:
            return self.datasets["resume_class_training"]
        if "real_pdf_resumes" in self.datasets:
            return self.datasets["real_pdf_resumes"]
        raise ValueError("No training dataset available")

    def get_skills_list(self) -> list:
        if "skills_list" in self.datasets:
            return self.datasets["skills_list"].iloc[:, 0].tolist()
        return []

    def get_job_role_category_map(self) -> dict:
        if "job_role_category_map" in self.datasets:
            df = self.datasets["job_role_category_map"]
            return dict(zip(df["Job Role"], df["Category"]))
        return {}