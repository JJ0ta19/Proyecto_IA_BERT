# Datos de Entrenamiento

## Función
Almacena el único dataset necesario para entrenar el modelo BERT de clasificación de currículums.

## Estructura

```
datos_entrenamiento/
└── 1_resume_classification/
    └── training_data.csv    # ÚNICO dataset usado
```

## Dataset

### training_data.csv
- **10,000 currículums** categorizados
- **42 categorías** profesionales
- Columnas:
  - `Resume Text`: Texto del currículum
  - `Category`: Categoría profesional

### Distribución de categorías
```
Technology                      2511 (25.1%)
Data & Analytics                 568 (5.7%)
Healthcare                       488 (4.9%)
Marketing & Sales                463 (4.6%)
Engineering & Manufacturing      435 (4.4%)
... (38 más categorías)
```

## Uso en el Código

```python
from src.datasets.data_loader import DatasetLoader

# Cargar dataset
loader = DatasetLoader()
df = loader.load_training_data()

# Obtener textos y etiquetas
texts, labels = loader.get_texts_and_labels()

# O directamente del DataFrame
texts = df['Resume Text'].tolist()
labels = df['Category'].tolist()
```

## Notas
- **Solo este dataset** es necesario para entrenar el modelo BERT
- El modelo se entrenó con:
  - 3 épocas
  - 80% entrenamiento, 20% validación
  - Batch size: 16
  - Learning rate: 2e-5
- Otros archivos CSV fueron eliminados del código (no se usaban)