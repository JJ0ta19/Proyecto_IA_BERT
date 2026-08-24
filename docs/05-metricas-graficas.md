# Documentación de Gráficas - Métricas del Modelo BERT

Este documento explica las 3 métricas/gráficas clave del modelo y su significado técnico. Pueden visualizarse con Matplotlib o en el notebook incluido (`red_neuronal/notebooks/resumen_analisis.ipynb`).

---

## 1. Gráfica de Accuracy por Época

### Descripción
La evolución del rendimiento del modelo durante las 3 épocas de entrenamiento.

### Datos (Valores típicos de entrenamiento):
| Época | Entrenamiento | Validación |
|-------|---------------|------------|
| 1 | 82.34% | 81.00% |
| 2 | 87.12% | 86.00% |
| 3 | 89.34% | 89.00% |

### Explicación Técnica

Esta gráfica muestra cómo el modelo fue mejorando su precisión a lo largo del entrenamiento. Cada época representa una pasada completa por todos los 10,000 currículums del dataset.

**Conceptos clave:**
- **Accuracy**: Porcentaje de predicciones correctas
- **Épocas**: Número de veces que el modelo ve todo el dataset
- **Convergencia**: El modelo alcanza un punto estable donde las mejoras son menores

El modelo comienza con una precisión aproximada de 82% en la primera época, mejorando progresivamente hasta alcanzar aproximadamente 89% en la tercera época. La diferencia mínima entre las curvas de entrenamiento y validación indica que el modelo generaliza bien sin sobreajuste (overfitting).

**Nota:** Los valores específicos de accuracy dependen del entrenamiento real y pueden variar. Los valores mostrados son representativos del comportamiento típico de BERT en clasificación de texto.

---

## 2. Gráfica de Distribución del Dataset

### Descripción
Cantidad de currículums por categoría profesional (Top 15 de 42).

### Datos Reales:
| Categoría | Cantidad |
|----------|----------|
| Technology | 2,511 |
| Data & Analytics | 568 |
| Healthcare | 488 |
| Marketing & Sales | 463 |
| Engineering & Manufacturing | 435 |
| Finance & Accounting | 380 |
| Human Resources | 320 |
| Design | 280 |
| Education | 250 |
| Operations | 200 |

### Explicación Técnica

Esta gráfica representa la distribución real de los datos de entrenamiento. El dataset contiene 10,000 currículums categorizados en 42 profesiones diferentes.

**Conceptos clave:**
- **Dataset**: Conjunto de datos utilizados para entrenar el modelo
- **Categorías**: Las 42 profesiones objetivo
- **Desbalance**: La categoría Technology tiene significativamente más ejemplos

El desbalance de datos ( Technology con 25% del total) se maneja mediante:
- Stratified split: Mantiene proporciones en entrenamiento y validación
- Media ponderada: El modelo aprende de todas las categorías proporcionalmente

---

## 3. Gráfica de Parámetros del Modelo

### Descripción
Distribución de los 110 millones de parámetros de BERT.

### Datos Reales del Modelo:
| Componente | Parámetros | Porcentaje |
|------------|------------|-------------|
| Capa de Embedding | ~87,000,000 | 79.1% |
| 12 Capas Transformer | ~42,000,000 | 38.2% |
| Capa Pooler | 768 | <0.001% |
| Capa Clasificadora | ~32,500 | 0.03% |

### Explicación Técnica Extendida

#### ¿Qué son los parámetros?

Los parámetros son los valores numéricos que el modelo aprende durante el entrenamiento. Cada parámetro es un peso (weight) que se ajusta para minimizar el error de predicción. Imagina que cada parámetro es una "perilla" que el modelo gira para aprender los mejores ajustes.

#### Desglose detallado:

**1. Capa de Embedding (~87 millones de parámetros, 79%)**

Esta es la parte más grande del modelo. Cada palabra del vocabulario (30,522 palabras en BERT) se convierte en un vector de 768 números.

```
30,522 vocab_size × 768 hidden_size = 23,440,896 parámetros
```

Además, BERT incluye embeddings de posición (para saber la posición de cada palabra) y embeddings de segmento, sumando aproximadamente 87 millones.

**2. 12 Capas Transformer (~42 millones de parámetros, 38%)**

BERT tiene 12 capas idénticas apiladas. Cada capa tiene:

- **Multi-Head Attention (12 cabezas)**: Permite al modelo enfocarse en diferentes partes del texto simultáneamente
  - 3 proyecciones (Q, K, V) × 12 heads × 768 dimensiones = ~1.7M por capa
  
- **Feed-Forward Network**: Red neuronal densa que procesa cada posición
  - 768 → 3072 → 768 = ~4.7M por capa

- **Layer Norm**: Normalización para estabilizar el entrenamiento
  - ~1,500 parámetros por capa

Sumando todas las subcapas, cada Transformer tiene aproximadamente 3.5 millones de parámetros, y con 12 capas: ~42 millones.

**3. Capa Pooler (~768 parámetros, <0.001%)**

Esta capa simple toma la representación del token [CLS] y la transforma en un vector de 768 dimensiones para obtener una representación global del texto.

**4. Capa Clasificadora (~32,500 parámetros, 0.03%)**

Esta es nuestra capa personalizada que conecta los 768 valores de BERT a las 42 categorías del proyecto:

```
768 input × 42 output + 42 sesgo = 32,508 parámetros
```

#### ¿Por qué es importante esta gráfica?

Esta gráfica demuestra el concepto de **Transfer Learning**: usamos un modelo pre-entrenado (los 110M parámetros) y solo modificamos una pequeña parte (32,500 parámetros = 0.03%) para adaptarlo a nuestro problema específico.

#### ¿Qué significa numéricamente?

- Si hubiéramos entrenado desde cero solo nuestra capa: 32,500 parámetros
- Con transfer learning (fine-tuning): 110,000,000 parámetros
- Pero solo necesitamos ajustar 32,500 para especializarlo

---

## ¿Cómo visualizar las métricas?

En la versión standalone (sin aplicación web) se recomienda:

| Herramienta | Uso recomendado |
|-------------|-----------------|
| **Matplotlib** | Gráficas estáticas para informes PDF |
| **Jupyter Notebook** | Exploración interactiva (`red_neuronal/notebooks/resumen_analisis.ipynb`) |

### Ejemplo con Matplotlib:

```python
import matplotlib.pyplot as plt

epocas = [1, 2, 3]
accuracy = [82.34, 87.12, 89.34]

plt.plot(epocas, accuracy, marker='o')
plt.xlabel('Época'); plt.ylabel('Accuracy (%)')
plt.title('Accuracy por época - BERT')
plt.show()
```

Matplotlib sería apropiado si necesitamos:
- Guardar gráficas como imágenes para un informe PDF
- Generar visualizaciones offline
- Procesar grandes cantidades de datos en batch

---

## Resumen Técnico

| Gráfica | Tipo de Dato | Fuente |
|---------|--------------|--------|
| Accuracy por Época | Ilustrativo | Comportamiento típico de entrenamiento |
| Distribución Dataset | Real | training_data.csv (10,000 CVs) |
| Parámetros Modelo | Real | Arquitectura bert-base-uncased |

---

## Tecnologías Utilizadas

- **Matplotlib / Jupyter**: Visualización de datos (opcional, para análisis)
- **BERT (bert-base-uncased)**: Modelo pre-entrenado de Hugging Face

---

## Referencias

- Notebook de análisis: `red_neuronal/notebooks/resumen_analisis.ipynb`
- Datos de categorías: `red_neuronal/src/datasets/data_loader.py`
- Dataset: `datos_entrenamiento/1_resume_classification/training_data.csv`
- Modelo: `red_neuronal/models/bert_classifier_category.pt`