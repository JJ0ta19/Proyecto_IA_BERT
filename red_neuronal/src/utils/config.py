"""
=================================================================
CONFIGURACIÓN DEL PROYECTO
Universidad - Variables Globales y Rutas

Este módulo contiene todas las constantes y configuraciones
del proyecto:

1. Rutas de directorios
2. Rutas de datasets
3. Parámetros del modelo BERT
4. Configuración de entrenamiento

Estas variables se importan desde otros módulos para
mantener configuración centralizada.

=================================================================
"""

import os  # Operaciones del sistema de archivos


# ====================================================================
# RUTAS BASE
# ====================================================================

# BASE_DIR: Ruta al directorio raíz del proyecto (red_neuronal)
# Obtiene la ruta absoluta del archivo actual y sube dos niveles
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATASET_DIR: Ruta a la carpeta de datasets
# Sube desde red_neuronal/src/utils -> red_neuronal -> proyecto -> datos_entrenamiento
DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "datos_entrenamiento")

# ====================================================================
# RUTAS DE DATASETS
# ====================================================================

# Diccionario con rutas a todos los datasets
# Cada ключ (key) es un identificador, cada value es la ruta al CSV
DATASET_PATHS = {
    # Dataset 1: Clasificación de currículums
    "1_resume_classification_training": os.path.join(DATASET_DIR, "1_resume_classification", "training_data.csv"),
    "1_resume_classification_jobs": os.path.join(DATASET_DIR, "1_resume_classification", "job_roles.csv"),
    "1_resume_classification_skills": os.path.join(DATASET_DIR, "1_resume_classification", "skills_list.csv"),
    
    # Dataset 2: PDFs de currículums reales (ya eliminado)
    "2_real_pdf_resumes": os.path.join(DATASET_DIR, "2_real_pdf_resumes", "Resume", "Resume.csv"),
    
    # Dataset 3: Matching job-currículum
    "3_job_matching": os.path.join(DATASET_DIR, "3_job_resume_matching", "job_resume_fit.csv"),
    
    # Dataset 5: Taxonomía de skills
    "5_skills": os.path.join(DATASET_DIR, "5_skills_taxonomy", "skills_en.csv"),
    "5_occupations": os.path.join(DATASET_DIR, "5_skills_taxonomy", "occupations_en.csv"),
    "5_skill_relations": os.path.join(DATASET_DIR, "5_skills_taxonomy", "occupationSkillRelations.csv"),
}


# ====================================================================
# RUTAS DE MODELOS
# ====================================================================

# Directorio donde se guarda el modelo entrenado
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ====================================================================
# PARÁMETROS DE ENTRENAMIENTO
# ====================================================================

# Longitud máxima de secuencia para BERT (512 tokens = máximo de BERT)
MAX_LEN = 512

# Tamaño de batch (número de samples por iteración)
BATCH_SIZE = 16

# Número de épocas (cuántas veces pasa por todo el dataset)
EPOCHS = 3

# Learning rate (tasa de aprendizaje - muy pequeño para BERT)
LEARNING_RATE = 2e-5  # 0.00002

# Número de workers para DataLoader (procesamiento paralelo)
NUM_WORKERS = 4


# ====================================================================
# MODELOS PRE-ENTRENADOS
# ====================================================================

# Modelo BERT a usar: bert-base-uncased
# - 12 capas transformer
# - 768 hidden dimensions
# - 12 attention heads
# - 110M parámetros
# - Case insensitive (no diferencia mayúsculas/minúsculas)
BERT_MODEL = "bert-base-uncased"

# Modelo de sentence-transformers para embeddings
# all-MiniLM-L6-v2: pequeño y rápido (22M parámetros)
# Útil para búsqueda de similitud semántica
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"


# ====================================================================
# DISPOSITIVO
# ====================================================================

# Seleccionar dispositivo automáticamente
# Si se ejecuta como script principal -> CPU
# Si se importa como módulo -> GPU si está disponible, sino CPU
DEVICE = "cuda" if __name__ != "__main__" else "cpu"


# ====================================================================
# SEMILLA ALEATORIA
# ====================================================================

# Semilla para reproducibilidad de resultados
RANDOM_SEED = 42


