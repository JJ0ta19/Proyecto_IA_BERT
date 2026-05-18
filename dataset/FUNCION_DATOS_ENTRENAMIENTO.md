# Datos de Entrenamiento

## Función
Almacena los datos utilizados para entrenar el modelo de clasificación de currículums.

## Contenido

```
datos_entrenamiento/
└── 1_resume_classification/
    └── training_data.csv
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
Technology                      2511
Data & Analytics                 568
Healthcare                       488
Marketing & Sales                463
Engineering & Manufacturing      435
... (38 más)
```

## Uso
```python
import pandas as pd

df = pd.read_csv('training_data.csv')
texts = df['Resume Text'].tolist()
labels = df['Category'].tolist()
```

## Notas
- Este es el único dataset necesario para entrenar el modelo
- El modelo se entrenó con 3 épocas
- 80% entrenamiento, 20% validación