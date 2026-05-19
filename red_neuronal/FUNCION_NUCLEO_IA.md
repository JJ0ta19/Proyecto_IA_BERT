# Núcleo de Inteligencia Artificial

## Función
Contiene todo el código de procesamiento de IA, modelo BERT, preprocesamiento NLP y extracción de texto.

## Estructura

```
nucleo_ia/
├── src/
│   ├── models/
│   │   └── bert_classifier.py    # Modelo BERT para clasificación
│   ├── datasets/
│   │   ├── data_loader.py        # Carga datasets
│   │   └── preprocessor.py       # Preprocesa datos para BERT
│   ├── preprocessing/
│   │   ├── text_cleaner.py        # Limpia texto
│   │   ├── skill_extractor.py    # Extrae skills
│   │   └── nlp_processor.py      # Pipeline NLP completo (NER, lemmatización)
│   └── utils/
│       └── config.py             # Configuración global
├── models/
│   └── bert_classifier_category.pt  # Modelo entrenado
├── notebooks/
│   └── resumen_analisis.ipynb    # Jupyter notebook para pruebas
├── main.py                       # Script principal
└── requirements.txt              # Dependencias
```

## Componentes

### Modelo BERT
- **BertClassifierModel**: Clasificador basado en BERT
- **BertClassifier**: Arquitectura personalizada
- **ResumeDataset**: Dataset para PyTorch

### Preprocesamiento
1. **TextCleaner**: Limpia texto (elimina caracteres especiales, stopwords)
2. **NLPProcessor**: Procesamiento completo (NER, lemmatización, extracción de secciones)
3. **SkillExtractor**: Detecta habilidades técnicas y blandas

### Extracción de Texto
- **OCR con Tesseract**: Para PDFs escaneados
- **PyMuPDF**: Para PDFs con texto

## Datos de Entrenamiento
El modelo fue entrenado con `training_data.csv` (10,000 currículums categorizados).

## Uso
```python
from src.models.bert_classifier import BertClassifierModel

classifier = BertClassifierModel(num_classes=42, device='cpu')
classifier.load_model('models/bert_classifier_category.pt')
predictions = classifier.predict(['texto de ejemplo'])
```

## Requerimientos
- torch >= 2.0.0
- transformers >= 4.30.0
- spacy >= 3.5.0
- pytesseract >= 0.3.10
- deep-translator >= 1.11.0