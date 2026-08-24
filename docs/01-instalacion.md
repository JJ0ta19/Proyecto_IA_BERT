# Guía de Instalación y Configuración

Pasos para instalar las dependencias e inicializar el sistema de clasificación de currículums.

---

## Requisitos Previos

| Requisito | Versión | Notas |
|-----------|---------|-------|
| Python | 3.11 (recomendado) | `runtime.txt` define 3.11 para despliegue |
| pip | Última | Gestor de paquetes de Python |
| Tesseract OCR | 5.x | Necesario solo si se procesan PDFs escaneados |
| Git | Opcional | Para clonar el repositorio |

---

## 1. Estructura del Proyecto

```
Proyecto_IA_BERT/
├── manage.py                  # Punto de entrada Django
├── proyecto_cv/               # Configuración del proyecto Django
├── aplicacion_web/            # Aplicación web (vistas, URLs, templates)
├── red_neuronal/              # Modelo BERT y NLP
│   ├── src/                   # Código fuente del modelo
│   ├── models/                # Artefactos del modelo (generado en runtime)
│   └── requirements.txt
├── datos_entrenamiento/       # Dataset (training_data.csv)
├── pdfs_temporales/           # PDFs subidos por usuarios (se eliminan solos)
├── build.sh / render.yaml     # Despliegue en Render
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

### 2.2 Dependencias del proyecto

```bash
# 1. Framework web
pip install "Django>=4.0"

# 2. Red neuronal (torch, transformers, spacy, OCR, etc.)
cd red_neuronal
pip install -r requirements.txt
```

Las dependencias principales son: `torch`, `transformers`, `pandas`, `scikit-learn`, `spacy`, `pytesseract`, `deep-translator`, `langdetect`, `pymupdf`, `Pillow`. Ver [02-arquitectura-tecnologias.md](02-arquitectura-tecnologias.md) para el rol de cada una.

### 2.3 Descargar modelo de spaCy

```bash
python -m spacy download en_core_web_sm
```

### 2.4 Instalar Tesseract OCR (solo PDFs escaneados)

**Windows:**
1. Descargar desde: <https://github.com/UB-Mannheim/tesseract/wiki>
2. Instalar y agregar al PATH del sistema
3. Opcional: instalar datos de idioma español (`tesseract-ocr-spa`)

**Linux (Ubuntu):**
```bash
sudo apt-get update && sudo apt-get install tesseract-ocr tesseract-ocr-spa
```

**Mac (Homebrew):**
```bash
brew install tesseract
```

---

## 3. Configurar el Proyecto Django

El proyecto ya viene configurado; estos pasos son de verificación:

### 3.1 settings.py (`proyecto_cv/settings.py`)

```python
INSTALLED_APPS = [..., 'aplicacion_web']

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'pdfs_temporales'

# Ruta del núcleo de IA al PYTHONPATH
sys.path.insert(0, os.path.join(BASE_DIR, 'red_neuronal'))
```

### 3.2 Crear carpetas generadas en runtime

```bash
mkdir pdfs_temporales          # PDFs temporales de usuarios
mkdir red_neuronal/models      # Artefacto del modelo entrenado
```

---

## 4. Ejecutar el Proyecto

```bash
# Migraciones
python manage.py makemigrations
python manage.py migrate

# Servidor de desarrollo
python manage.py runserver
```

La aplicación queda disponible en: <http://127.0.0.1:8000/>

> **Nota PowerShell:** si Django no encuentra el módulo de settings, ejecutar antes:
> `$env:DJANGO_SETTINGS_MODULE="proyecto_cv.settings"`

---

## 5. Modelo BERT

Al iniciar, la aplicación verifica si existe `red_neuronal/models/bert_classifier_category.pt`:

1. **Si existe** → se carga directamente.
2. **Si NO existe** → se descarga automáticamente desde Google Drive usando `gdown` (implementado en `aplicacion_web/views.py`).

### Descarga manual (alternativa)

**Enlace:** <https://drive.google.com/file/d/1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um/view?usp=sharing>

Guardar como: `red_neuronal/models/bert_classifier_category.pt`

### Entrenamiento local (opcional)

```bash
cd red_neuronal
python train_bert_classifier.py
```

Requiere el dataset en `datos_entrenamiento/1_resume_classification/training_data.csv`. Ver [05-datos-entrenamiento.md](05-datos-entrenamiento.md).

---

## 6. Verificar la Instalación

1. Abrir <http://127.0.0.1:8000/>
2. Subir un PDF de currículum
3. El sistema debe mostrar la categoría profesional predicha y su confianza

---

## 7. Solución de Problemas

### `ModuleNotFoundError: No module named 'src'`

Agregar la ruta del núcleo de IA al `PYTHONPATH` (ya incluida en `settings.py`):

```python
import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'red_neuronal'))
```

### `TesseractNotFoundError`

Configurar la ruta de Tesseract en `views.py`:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### `No such file or directory: training_data.csv`

Verificar que el dataset exista en `datos_entrenamiento/1_resume_classification/training_data.csv`.

---

## 8. Resumen de Comandos

```bash
venv\Scripts\activate                 # Activar entorno (Windows)
pip install "Django>=4.0"
cd red_neuronal && pip install -r requirements.txt
python -m spacy download en_core_web_sm
python manage.py makemigrations && python manage.py migrate
python manage.py runserver
```
