# Aplicación Web Django

## Función
Interfaz web del proyecto. Permite a los usuarios subir currículums en formato PDF y obtener la predicción de categoría profesional usando el modelo BERT.

## Contenido

### Vistas (views.py)
- `home()` - Página principal
- `upload_cv()` - Procesa PDF desde la página principal
- `analyze_pdf()` - Procesa PDF desde la página de análisis
- `load_classifier()` - Carga el modelo BERT
- `predict_category()` - Realiza la predicción
- `extract_text_from_pdf()` - Extrae texto del PDF usando OCR
- `translate_to_english()` - Traduce texto al inglés

### URLs (urls.py)
Rutas disponibles:
- `/` - Página principal
- `/analizar/` - Página de análisis avanzado
- `/info/` - Información del modelo

### Templates
- `base.html` - Plantilla base con estilos
- `index.html` - Página principal con formulario
- `analyze.html` - Página de análisis con más detalles

## Flujo
1. Usuario sube PDF → Se guarda en media/
2. Se extrae texto con OCR (Tesseract)
3. Se preprocesa texto (traducción, NLP, NER)
4. Se pasa al modelo BERT → Predicción
5. Se muestra resultado en la vista
6. PDF temporal se elimina

## Requerimientos
- Django 4.2+
- Modelo BERT entrenado (ubicado en nucleo_ia/models/)