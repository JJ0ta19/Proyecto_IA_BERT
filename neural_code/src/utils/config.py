import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "dataset")

DATASET_PATHS = {
    "1_resume_classification_training": os.path.join(DATASET_DIR, "1_resume_classification", "training_data.csv"),
    "1_resume_classification_jobs": os.path.join(DATASET_DIR, "1_resume_classification", "job_roles.csv"),
    "1_resume_classification_skills": os.path.join(DATASET_DIR, "1_resume_classification", "skills_list.csv"),
    "2_real_pdf_resumes": os.path.join(DATASET_DIR, "2_real_pdf_resumes", "Resume", "Resume.csv"),
    "3_job_matching": os.path.join(DATASET_DIR, "3_job_resume_matching", "job_resume_fit.csv"),
    "5_skills": os.path.join(DATASET_DIR, "5_skills_taxonomy", "skills_en.csv"),
    "5_occupations": os.path.join(DATASET_DIR, "5_skills_taxonomy", "occupations_en.csv"),
    "5_skill_relations": os.path.join(DATASET_DIR, "5_skills_taxonomy", "occupationSkillRelations.csv"),
}

MODEL_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(MODEL_DIR, "output")

MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
NUM_WORKERS = 4

BERT_MODEL = "bert-base-uncased"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

DEVICE = "cuda" if __name__ != "__main__" else "cpu"

RANDOM_SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)