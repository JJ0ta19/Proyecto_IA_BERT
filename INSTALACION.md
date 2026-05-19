# Guía de Instalación y Configuración del Proyecto

Este documento contiene los pasos para instalar las dependencias e inicializar el proyecto Django.

---

## Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git (opcional)

---

## 1. Estructura del Proyecto

```
Proyecto_IA_BERT/
├── aplicacion_web/        # Aplicación Django
│   ├── templates/         # Plantillas HTML
│   ├── views.py          # Vistas de Django
│   ├── urls.py           # Rutas URL
│   └── ...
├── red_neuronal/         # Modelo BERT y NLP
│   ├── src/
│   ├── models/
│   └── requirements.txt
├── datos_entrenamiento/  # Dataset
└── pdfs_temporales/      # PDFs subidos por usuarios
```

---

## 2. Instalar Dependencias

### 2.1 Crear entorno virtual (recomendado)

```bash
# En Windows
python -m venv venv

# Activar entorno virtual
venv\Scripts\activate

# En Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2.2 Instalar dependencias del proyecto

El proyecto tiene dos archivos de requirements:

```bash
# 1. Dependencias principales (Django, etc.)
pip install Django>=4.0

# 2. Dependencias de la red neuronal
cd red_neuronal
pip install -r requirements.txt

# Las principales dependencias son:
pip install torch transformers pandas scikit-learn
pip install spacy pytesseract deep-translator langdetect
pip install pymupdf pillow
```

### 2.3 Descargar modelo de spaCy

```bash
python -m spacy download en_core_web_sm
```

### 2.4 Instalar Tesseract OCR

**Windows:**
1. Descargar Tesseract desde: https://github.com/UB-Mannheim/tesseract/wiki
2. Instalar y agregar al PATH del sistema
3. Opcional: descargar datos de idioma español

**Linux (Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-spa  # Español
```

**Mac (Homebrew):**
```bash
brew install tesseract
```

---

## 3. Inicializar Proyecto Django

### 3.1 Crear proyecto Django (si no existe)

```bash
cd C:\Users\WinterOS\Documents\IA\TALLER\Proyecto grupo 5\PRODUCCION\Proyecto_IA_BERT

# Crear proyecto Django
django-admin startproject aplicacion_web .

# O si prefieres un nombre diferente:
django-admin startproject mi_proyecto
```

### 3.2 Estructura después de crear Django

```
aplicacion_web/
├── manage.py              # Punto de entrada Django
├── aplicacion_web/       # Configuración del proyecto
│   ├── __init__.py
│   ├── settings.py       # Configuración principal
│   ├── urls.py          # URLs del proyecto
│   └── wsgi.py
├── templates/            # Plantillas HTML
└── ...
```

### 3.3 Configurar settings.py

Editar `aplicacion_web/settings.py`:

```python
# Agregar al inicio del archivo
import os
from pathlib import Path

# Agregar aplicaciones instaladas
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'aplicacion_web',  # Tu aplicación
]

# Configurar rutas de templates
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        ...
    },
]

# Agregar configuración de medios
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'pdfs_temporales'

# Configurar rutas absolutas
import sys
sys.path.insert(0, os.path.join(BASE_DIR, 'red_neuronal'))
```

### 3.4 Configurar URLs

Editar `aplicacion_web/urls.py`:

```python
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('aplicacion_web.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

### 3.5 Crear carpetas necesarias

```bash
# Crear carpetas para PDFs temporales y modelos
mkdir -p pdfs_temporales
mkdir -p red_neuronal/models
```

---

## 4. Ejecutar el Proyecto

### 4.1 Hacer migraciones

```bash
python manage.py makemigrations
python manage.py migrate
```

### 4.2 Crear superusuario (opcional)

```bash
python manage.py createsuperuser
```

### 4.3 Iniciar servidor

```bash
# Windows PowerShell - ejecutar antes de cada comando
$env:DJANGO_SETTINGS_MODULE="proyecto_cv.settings"

# Luego ejecutar:
python manage.py runserver
```

El proyecto estará disponible en: http://127.0.0.1:8000/

---

## 5. Verificar Instalación

### 5.1 Verificar que el modelo BERT existe

Si el modelo no existe, entrenarlo primero:

```bash
cd red_neuronal
python train_bert_classifier.py
```

Esto creará: `red_neuronal/models/bert_classifier_category.pt`

### Alternativa: Descargar modelo pre-entrenado

Si no quieres entrenar el modelo, puedes descargarlo desde Google Drive:

**Enlace:** https://drive.google.com/file/d/1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um/view?usp=sharing

1. Descargar `bert_classifier_category.pt`
2. Guardar en: `red_neuronal/models/bert_classifier_category.pt`

### 5.2 Probar la aplicación

1. Abrir navegador en http://127.0.0.1:8000/
2. Subir un archivo PDF de currículum
3. El sistema debería clasificar el CV

---

## 6. Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'src'"

Agregar la ruta del proyecto al PYTHONPATH:

```python
# En settings.py o al inicio de views.py
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'red_neuronal'))
```

### Error: "TesseractNotFound"

Configurar la ruta de Tesseract en `views.py`:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Error: "No such file or directory: training_data.csv"

Verificar que el dataset existe en:
```
datos_entrenamiento/1_resume_classification/training_data.csv
```

---

## 7. Resumen de Comandos

```bash
# Activar entorno virtual (Windows)
venv\Scripts\activate

# Instalar dependencias
pip install Django torch transformers spacy pytesseract

# Descargar spaCy
python -m spacy download en_core_web_sm

# Inicializar Django
python manage.py makemigrations
python manage.py migrate

# Ejecutar servidor
python manage.py runserver
```

---

## Notas

- El proyecto usa **BERT** para clasificación de currículums
- Solo se necesita **1 dataset**: `training_data.csv` (10,000 CVs)
- El modelo debe estar en: `red_neuronal/models/bert_classifier_category.pt`
- La aplicación web está en la carpeta `aplicacion_web/`