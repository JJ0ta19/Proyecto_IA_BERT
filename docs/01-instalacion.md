# Guía de Instalación y Configuración

Pasos para instalar las dependencias y ejecutar el núcleo de IA del sistema de clasificación de currículums.

> **Rama `solo-red-neuronal`:** solo se requiere el entorno de Machine Learning; no hay aplicación web ni despliegue.

---

## Requisitos Previos

| Requisito | Versión | Notas |
|-----------|---------|-------|
| Python | 3.11 (recomendado) | Versión usada para entrenar el modelo |
| pip | Última | Gestor de paquetes de Python |
| Git | Opcional | Para clonar el repositorio |

---

## 1. Estructura del Proyecto

```
Proyecto_IA_BERT/
├── red_neuronal/              # Modelo BERT y NLP
│   ├── src/                   # Código fuente del modelo
│   ├── models/                # Artefactos del modelo (generado en runtime)
│   └── requirements.txt
├── datos_entrenamiento/       # Dataset (training_data.csv)
└── docs/                      # Documentación del proyecto
```

---

## 2. Instalar Dependencias

### 2.1 Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2.2 Dependencias del núcleo de IA

```bash
cd red_neuronal
pip install -r requirements.txt
```

Las principales: `torch`, `transformers`, `pandas`, `numpy`, `scikit-learn`, `spacy`, `gdown`, `tqdm`. Ver [02-arquitectura-tecnologias.md](02-arquitectura-tecnologias.md) para el rol de cada una.

### 2.3 Descargar modelo de spaCy

```bash
python -m spacy download en_core_web_sm
```

---

## 3. Preparar el Modelo BERT

Al ejecutar la predicción, el sistema verifica si existe `red_neuronal/models/bert_classifier_category.pt`:

1. **Si existe** → se carga directamente.
2. **Si NO existe** → se descarga automáticamente desde Google Drive usando `gdown`.

### Descarga manual (alternativa)

**Enlace:** <https://drive.google.com/file/d/1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um/view?usp=sharing>

Guardar como: `red_neuronal/models/bert_classifier_category.pt`

### Entrenamiento local (opcional)

```bash
cd red_neuronal
python train_bert_classifier.py
```

Requiere el dataset en `datos_entrenamiento/1_resume_classification/training_data.csv`. Ver [04-datos-entrenamiento.md](04-datos-entrenamiento.md).

---

## 4. Ejecutar

```bash
cd red_neuronal

# Predicción de ejemplo con el modelo entrenado
python predict.py

# Punto de entrada general
python main.py
```

---

## 5. Verificar la Instalación

1. Confirmar que `red_neuronal/models/bert_classifier_category.pt` existe (o que hay conexión a internet para descargarlo)
2. Ejecutar `python predict.py` dentro de `red_neuronal/`
3. El sistema debe mostrar la categoría profesional predicha y su confianza

---

## 6. Solución de Problemas

### `ModuleNotFoundError: No module named 'src'`

Ejecutar los scripts desde dentro de la carpeta `red_neuronal/`:

```bash
cd red_neuronal
python predict.py
```

### Error al instalar torch

Instalar la versión CPU-only desde el índice oficial:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### `No such file or directory: training_data.csv`

Verificar que el dataset exista en `datos_entrenamiento/1_resume_classification/training_data.csv`.

---

## 7. Resumen de Comandos

```bash
venv\Scripts\activate                 # Activar entorno (Windows)
cd red_neuronal
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python train_bert_classifier.py       # Entrenar
python predict.py                     # Predecir
```
