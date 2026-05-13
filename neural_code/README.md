# Sistema Inteligente de Análisis de Hojas de Vida

Pipeline completo de NLP y Deep Learning para análisis, clasificación y recomendación de hojas de vida.

---

## Índice

1. [Arquitectura del Sistema](#arquitectura-del-sistema)
2. [Estructura de Carpetas](#estructura-de-carpetas)
3. [Cómo Ejecutar](#cómo-ejecutar)
4. [Explicación de cada Módulo](#explicación-de-cada-módulo)
5. [Datasets Utilizados](#datasets-utilizados)
6. [Flujo del Pipeline](#flujo-del-pipeline)
7. [Próximos Pasos](#próximos-pasos)

---

## Arquitectura del Sistema

```
PDF Resume
    ↓
Text Extraction (PDF→texto)
    ↓
NLP Cleaning (limpieza)
    ↓
Skill Extraction (detectar habilidades)
    ↓
Embeddings (representación semántica)
    ↓
Resume Classification (clasificar profesión)
    ↓
Semantic Matching (matching con ofertas)
    ↓
Job Recommendation (recomendar trabajos)
```

---

## Estructura de Carpetas

```
neural_code/
├── main.py                      # Punto de entrada
├── requirements.txt             # Dependencias
│
├── src/
│   ├── preprocessing/          # Limpieza de texto y extracción de skills
│   │   ├── text_cleaner.py      # Limpiador de texto NLP
│   │   └── skill_extractor.py   # Detector de habilidades
│   │
│   ├── datasets/                # Carga y preprocesamiento de datos
│   │   ├── data_loader.py       # Carga todos los datasets
│   │   └── preprocessor.py      # Prepara datos para BERT
│   │
│   ├── models/                  # Modelos de Machine Learning
│   │   ├── bert_classifier.py  # Clasificador BERT
│   │   └── embedding_model.py  # SentenceTransformers
│   │
│   ├── matching/                # Sistema de matching laboral
│   │   └── job_matcher.py       # Matching semántico CV↔ofertas
│   │
│   ├── recommendation/         # Sistema de recomendaciones
│   │   └── recommender.py       # Recomendaciones de carrera
│   │
│   └── utils/                   # Configuración
│       └── config.py            # Rutas y parámetros
│
└── models/                      # Modelos entrenados (se crea al ejecutar)
```

---

## Cómo Ejecutar

### 1. Instalar Dependencias

```bash
cd neural_code
pip install -r requirements.txt
```

### 2. Ejecutar el Pipeline Completo

```bash
python main.py
```

Esto ejecutará:
- Carga de datasets
- Inicialización del modelo de embeddings
- Análisis de un CV de ejemplo

### 3. Entrenar el Clasificador BERT

```python
from main import ResumeAnalysisSystem

system = ResumeAnalysisSystem(use_gpu=False)
system.initialize()
system.train_classifier(epochs=3, batch_size=16)
```

### 4. Usar el Sistema para Analizar un CV

```python
from main import ResumeAnalysisSystem

system = ResumeAnalysisSystem(use_gpu=False)
system.initialize()

resume_text = """
EXPERIENCE
Software Engineer at Tech Corp
- Developed Python applications using Django and Flask
- Worked with PostgreSQL and MongoDB
- Deployed using AWS and Docker

SKILLS
Python, JavaScript, Django, AWS, Docker

EDUCATION
BS Computer Science
"""

result = system.analyze_resume(resume_text)
print(result)
```

---

## Explicación de cada Módulo

### 1. src/preprocessing/text_cleaner.py

**Qué hace:**
- Convierte texto a minúsculas
- Elimina caracteres especiales
- Expande contracciones (don't → do not)
- Normaliza espacios
- Elimina acentos

```python
cleaner = TextCleaner()
texto_limpio = cleaner.clean("Don't use Python! 🐍")
# Resultado: "do not use python"
```

### 2. src/preprocessing/skill_extractor.py

**Qué hace:**
- Detecta skills técnicos (Python, Java, AWS, etc.)
- Identifica frameworks (React, Django, Flask)
- Encuentra databases (PostgreSQL, MongoDB)
- Detecta cloud services (AWS, Azure)
- Extrae soft skills (leadership, comunicación)
- Identifica idiomas (english, spanish)

```python
extractor = SkillExtractor(skills_list)
skills = extractor.extract_from_resume("I know Python and AWS")
# Resultado: {'technical': ['python', 'aws'], 'soft': [], 'languages': []}
```

### 3. src/datasets/data_loader.py

**Qué hace:**
- Carga los 4 datasets activos:
  - `1_resume_classification` → Dataset principal de entrenamiento
  - `2_real_pdf_resumes` → CVs reales para robustez
  - `3_job_resume_matching` → Para matching laboral
  - `5_skills_taxonomy` → Normalización de skills

```python
loader = DatasetLoader()
datasets = loader.load_all()
# Carga todos los datasets automáticamente
```

**Nota:** `4_resume_large_scale` NO se carga (reservado para futuro)

### 4. src/datasets/preprocessor.py

**Qué hace:**
- Elimina valores nulos
- Limpia texto con TextCleaner
- Elimina duplicados
- Codifica labels a números
- Divide datos train/validation

```python
preprocessor = DatasetPreprocessor()
texts, labels = preprocessor.prepare_classification_data(df, "Resume_str", "Category")
train_texts, val_texts, train_labels, val_labels = preprocessor.prepare_for_bert(texts, labels)
```

### 5. src/models/bert_classifier.py

**Qué hace:**
- Carga BERT pre-entrenado
- Añade capa de clasificación
- Fine-tuning con los datos del dataset
- Soporta GPU/CUDA

```python
classifier = BertClassifierModel(num_classes=24, device="cuda")
classifier.train(train_texts, train_labels, val_texts, val_labels, epochs=3)
predictions = classifier.predict(["texto de CV"])
```

### 6. src/models/embedding_model.py

**Qué hace:**
- Usa SentenceTransformers (all-MiniLM-L6-v2)
- Convierte texto a vectores numéricos
- Calcula similaridad semántica
- Permite búsqueda por similaridad

```python
embedding_model = EmbeddingModel()
embeddings = embedding_model.encode(["texto1", "texto2"])
similarity = embedding_model.compute_similarity("python developer", "software engineer")
```

### 7. src/matching/job_matcher.py

**Qué hace:**
- Embeddea las ofertas laborales
- Compara CV con ofertas usando similaridad coseno
- Genera score de compatibilidad (0-100%)

```python
matcher = JobMatcher(embedding_model)
matcher.fit(jobs_df, "job_description")
matches = matcher.match("CV text", top_k=5)
# Resultado: [{'job_index': 0, 'compatibility_score': 87.5, 'job_info': {...}}]
```

### 8. src/recommendation/recommender.py

**Qué hace:**
- Recomienda profesiones basadas en el CV
- Detecta skills faltantes
- Sugiere paths de carrera

```python
recommender = JobRecommender(embedding_model, occupations_df, skill_relations_df)
recommender.fit_occupations()
recommendations = recommender.recommend_jobs("Data Scientist", skills, top_k=5)
gaps = recommender.suggest_skill_gaps(skills, "Data Scientist")
```

### 9. main.py (ResumeAnalysisSystem)

**Qué hace:**
- Coordina todos los módulos
- Expone API simple de uso

```python
system = ResumeAnalysisSystem(use_gpu=False)
system.initialize()              # Carga datos y modelos
system.train_classifier()        # Entrena BERT
result = system.analyze_resume(cv_text)  # Analiza un CV
```

---

## Datasets Utilizados

| Dataset | Uso | Estado |
|---------|-----|--------|
| **1_resume_classification** | Entrenamiento principal | ✅ Activo |
| **2_real_pdf_resumes** | Robustez y datos reales | ✅ Activo |
| **3_job_resume_matching** | Matching laboral | ✅ Activo |
| **4_resume_large_scale** | Futuro | ⏸️ Reservado |
| **5_skills_taxonomy** | Normalización skills | ✅ Activo |

---

## Flujo del Pipeline

```
1. DATA LOADING
   DatasetLoader → carga 4 datasets activos

2. PREPROCESSING
   TextCleaner → limpia texto
   SkillExtractor → detecta habilidades
   DatasetPreprocessor → prepara datos

3. TRAINING
   BertClassifier → fine-tuning de BERT

4. EMBEDDINGS
   EmbeddingModel → SentenceTransformers

5. MATCHING
   JobMatcher → similaridad semántica

6. RECOMMENDATION
   JobRecommender → suggestions de carrera

7. INFERENCE
   analyze_resume() → análisis completo del CV
```

---

## Parámetros Configurables

En `src/utils/config.py`:

```python
MAX_LEN = 512          # Longitud máxima de tokens
BATCH_SIZE = 16        # Tamaño de batch
EPOCHS = 3            # Épocas de entrenamiento
LEARNING_RATE = 2e-5   # Learning rate para BERT
BERT_MODEL = "bert-base-uncased"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
```

---

## Próximos Pasos

1. **Ejecutar entrenamiento** con los datasets
2. **Guardar modelos entrenados**
3. **Crear API con FastAPI** para integrar con Django
4. **Crear frontend** para subir PDFs
5. **Añadir 4_resume_large_scale** para entrenamiento avanzado

---

## Requisitos

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
pandas>=1.5.0
scikit-learn>=1.2.0
numpy>=1.23.0
spacy>=3.5.0
faiss-cpu>=1.7.0
pymupdf>=1.22.0
tqdm>=4.65.0
```

---

## Notas

- El sistema funciona sin GPU (CPU)
- Para GPU, cambia `use_gpu=False` a `use_gpu=True` en `main.py`
- Los modelos se guardan en `models/checkpoints/`
- El dataset 4_resume_large_scale está preparado para uso futuro