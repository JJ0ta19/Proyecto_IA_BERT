import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = "D:/UNIVERSIDAD 2026A/PROYECTO IA/final/Proyecto_IA_BERT/data"

DATASET_PATHS = {
    "1_resume_classification_training": os.path.join(DATASET_DIR, "1_resume_classification", "training_data.csv"),
    "1_resume_classification_jobs": os.path.join(DATASET_DIR, "1_resume_classification", "job_roles.csv"),
    "1_resume_classification_skills": os.path.join(DATASET_DIR, "1_resume_classification", "skills_list.csv"),
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

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"[CONFIG] GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[CONFIG] GPU no disponible - usando CPU")

RANDOM_SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)