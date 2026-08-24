# Arquitectura y Tecnologías

Descripción técnica de las tecnologías del proyecto, su rol en el pipeline y la justificación de cada elección.

> **Rama `solo-red-neuronal`:** se describe únicamente el pipeline de Machine Learning. La versión web (Django) y el despliegue corresponden a la rama `Main`.

---

## 1. Pipeline del Núcleo de IA

```
Texto del currículum
    ↓
TextCleaner (limpieza de texto)
    ↓
spaCy (NER, lemmatización, extracción de secciones)
↓
SkillExtractor (detección de habilidades)
    ↓
BERT tokenizer (convertir texto a números)
    ↓
Modelo BERT (predicción de categoría)
    ↓
Categoría profesional + confianza %
```

---

## 2. Dependencias de IA y Procesamiento

### Deep Learning y Modelos de Lenguaje

**torch (PyTorch)** — Framework de deep learning de Meta. Permite entrenar redes neuronales mediante diferenciación automática y retropropagación. Usado para definir, entrenar y evaluar el modelo BERT.

**transformers (Hugging Face)** — Librería con arquitecturas pre-entrenadas de última generación (BERT, RoBERTa, DistilBERT). Incluye tokenizadores, configuraciones y métodos de fine-tuning. Es el estándar de la industria para NLP.

**gdown** — Descarga del modelo entrenado desde Google Drive cuando no existe localmente.

### Ciencia de Datos

**pandas** — Manipulación y análisis de datos con DataFrames optimizados. Carga y preprocesa el dataset de currículums.

**numpy** — Computación numérica: arrays multidimensionales y funciones matemáticas de alto rendimiento.

**scikit-learn** — ML clásico: métricas (accuracy, F1-score), validación cruzada, split de datos y evaluación.

### NLP y Texto

**spaCy** — Framework industrial de NLP:
- NER: reconocimiento de entidades (personas, organizaciones, fechas)
- POS tagging: categorías gramaticales
- Lemmatización: reducción de palabras a su raíz
- Dependency parsing: análisis sintáctico

Modelo usado: `en_core_web_sm`.

**tqdm** — Barras de progreso durante el entrenamiento.

---

## 3. Resumen: Rol por Tecnología

| Tecnología | Propósito |
|------------|-----------|
| torch | Framework de deep learning |
| BERT + transformers | Modelo de machine learning |
| spaCy | Procesamiento de lenguaje natural |
| pandas / scikit-learn | Manejo y evaluación del dataset |
| gdown | Descarga automática del modelo |

Cada tecnología cumple un rol específico en el pipeline: limpiar → analizar → tokenizar → clasificar.
