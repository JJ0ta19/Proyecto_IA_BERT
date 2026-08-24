# Proyecto IA BERT — Clasificador de Currículums

Sistema de **clasificación de currículums (CVs) en 42 categorías profesionales** usando un modelo de red neuronal **BERT** fine-tuned (`bert-base-uncased`). El núcleo del proyecto es la red neuronal: recibe el texto de un CV y predice su categoría profesional con nivel de confianza.

> **Rama `solo-red-neuronal`:** versión del proyecto centrada exclusivamente en el modelo de Machine Learning, sin aplicación web ni configuración de despliegue. Para la versión web completa, ver la rama `Main`.

## Características

- **Modelo:** BERT (`bert-base-uncased`, ~110M parámetros) fine-tuned para 42 clases
- **Framework ML:** PyTorch + Transformers (Hugging Face)
- **NLP:** spaCy (NER, lemmatización), limpieza y extracción de habilidades
- **Dataset:** 10,000 currículums en 42 categorías
- **Carga del modelo:** automática — descarga desde Google Drive si no existe
- **Notebook** de análisis incluido (`red_neuronal/notebooks/resumen_analisis.ipynb`)

## Estructura del proyecto

```
Proyecto_IA_BERT/
├── red_neuronal/                   # Núcleo de IA
│   ├── main.py                     # Punto de entrada
│   ├── predict.py                  # Predicción con modelo entrenado
│   ├── train_bert_classifier.py    # Entrenamiento
│   ├── test_train.py               # Pruebas de entrenamiento
│   ├── requirements.txt            # Dependencias
│   ├── models/                     # Artefacto .pt (generado/descargado)
│   ├── notebooks/                  # Jupyter notebook de análisis
│   └── src/
│       ├── models/bert_classifier.py   # Clasificador BERT
│       ├── datasets/                   # Carga y preprocesamiento del dataset
│       └── preprocessing/              # Limpieza NLP y extracción de skills
├── datos_entrenamiento/            # Dataset: training_data.csv (10,000 CVs)
└── docs/                           # Documentación completa
```

## Requisitos

- Python 3.11 recomendado
- Dependencias: `red_neuronal/requirements.txt`

## Instalación rápida

```bash
git clone https://github.com/JJ0ta19/Proyecto_IA_BERT.git
cd Proyecto_IA_BERT

# Entorno virtual
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux/Mac

# Dependencias
cd red_neuronal && pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Guía detallada: [docs/01-instalacion.md](docs/01-instalacion.md)

## Uso

### Entrenar el modelo

```bash
cd red_neuronal
python train_bert_classifier.py
```

Genera `red_neuronal/models/bert_classifier_category.pt`.

### Predecir categoría de un CV

```bash
cd red_neuronal
python predict.py
```

Si el modelo no existe, se descarga automáticamente desde Google Drive.

### Uso programático

```python
from src.models.bert_classifier import BertClassifierModel

classifier = BertClassifierModel(num_classes=42, device='cpu')
classifier.load_model('models/bert_classifier_category.pt')
predictions = classifier.predict(['texto del curriculum'])
```

## Modelo de Machine Learning

| Aspecto | Valor |
|---------|-------|
| Arquitectura | BERT-base-uncased (12 capas Transformer) |
| Parámetros | ~110 millones |
| Tarea | Clasificación de texto (multi-clase) |
| Categorías | 42 profesiones |
| Dataset | 10,000 currículums (`training_data.csv`) |
| Entrenamiento | 3 épocas · batch 16 · lr 2e-5 · split 80/20 |
| Precisión | ~89% en validación |

El modelo se entrena con transfer learning: se ajusta el BERT pre-entrenado con una capa clasificadora propia (768 → 42). Detalles completos en [docs/03-red-neuronal-bert.md](docs/03-red-neuronal-bert.md).

## Solución de problemas

| Error | Causa/Solución |
|-------|----------------|
| `ModuleNotFoundError: No module named 'src'` | Ejecutar los scripts dentro de `red_neuronal/` |
| `No such file: training_data.csv` | Verificar dataset en `datos_entrenamiento/` |
| Modelo no descarga | Descarga manual desde Google Drive (ver docs/01) |

Más detalles: [docs/01-instalacion.md](docs/01-instalacion.md)

## Documentación

| Documento | Contenido |
|-----------|-----------|
| [01-instalacion.md](docs/01-instalacion.md) | Instalación, configuración y solución de problemas |
| [02-arquitectura-tecnologias.md](docs/02-arquitectura-tecnologias.md) | Pipeline, tecnologías y justificación |
| [03-red-neuronal-bert.md](docs/03-red-neuronal-bert.md) | Documentación técnica completa del modelo BERT |
| [04-datos-entrenamiento.md](docs/04-datos-entrenamiento.md) | Dataset y distribución de categorías |
| [05-metricas-graficas.md](docs/05-metricas-graficas.md) | Métricas del modelo y gráficas |
| [documentacion-completa.html](docs/documentacion-completa.html) | Manual técnico completo (HTML) |
| [manual-tecnico.pdf](docs/manual-tecnico.pdf) | Manual técnico completo (PDF) |
