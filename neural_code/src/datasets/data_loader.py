import pandas as pd
import os
from typing import Dict, Tuple, Optional
from ..utils.config import DATASET_PATHS, DATASET_DIR

class DatasetLoader:
    def __init__(self):
        self.datasets = {}

    def load_all(self) -> Dict[str, pd.DataFrame]:
        print("Loading datasets...")
        self._load_resume_classification()
        self._load_real_pdf_resumes()
        self._load_job_matching()
        self._load_skills_taxonomy()
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
            print(f"  - training_data.csv: {len(df)} rows")

        if os.path.exists(jobs_path):
            df = pd.read_csv(jobs_path)
            self.datasets["job_roles"] = df
            print(f"  - job_roles.csv: {len(df)} rows")

        if os.path.exists(skills_path):
            df = pd.read_csv(skills_path)
            self.datasets["skills_list"] = df
            print(f"  - skills_list.csv: {len(df)} rows")

    def _load_real_pdf_resumes(self):
        print("Loading 2_real_pdf_resumes...")
        path = DATASET_PATHS["2_real_pdf_resumes"]
        if os.path.exists(path):
            df = pd.read_csv(path)
            self.datasets["real_pdf_resumes"] = df
            print(f"  - Resume.csv: {len(df)} rows, {len(df.columns)} columns")

    def _load_job_matching(self):
        print("Loading 3_job_resume_matching...")
        path = DATASET_PATHS["3_job_matching"]
        if os.path.exists(path):
            df = pd.read_csv(path)
            self.datasets["job_matching"] = df
            print(f"  - job_resume_fit.csv: {len(df)} rows")

    def _load_skills_taxonomy(self):
        print("Loading 5_skills_taxonomy...")
        skills_path = DATASET_PATHS["5_skills"]
        occupations_path = DATASET_PATHS["5_occupations"]
        relations_path = DATASET_PATHS["5_skill_relations"]

        if os.path.exists(skills_path):
            df = pd.read_csv(skills_path)
            self.datasets["skills_taxonomy"] = df
            print(f"  - skills_en.csv: {len(df)} rows")

        if os.path.exists(occupations_path):
            df = pd.read_csv(occupations_path)
            self.datasets["occupations"] = df
            print(f"  - occupations_en.csv: {len(df)} rows")

        if os.path.exists(relations_path):
            df = pd.read_csv(relations_path)
            self.datasets["skill_relations"] = df
            print(f"  - occupationSkillRelations.csv: {len(df)} rows")

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
        if "skills_taxonomy" in self.datasets:
            return self.datasets["skills_taxonomy"].iloc[:, 0].tolist()
        return []

    def get_job_matching_data(self) -> Optional[pd.DataFrame]:
        return self.datasets.get("job_matching")

    def get_occupations(self) -> Optional[pd.DataFrame]:
        return self.datasets.get("occupations")

    def get_skill_relations(self) -> Optional[pd.DataFrame]:
        return self.datasets.get("skill_relations")