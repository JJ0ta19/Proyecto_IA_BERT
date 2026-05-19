# Dependencias del Proyecto - Explicación Técnica

## deep learning y Modelos de Lenguaje

### torch (PyTorch)
Framework de deep learning de Facebook/Meta. Permite entrenar redes neuronales mediante diferenciación automática y retropropagación. Utilizado para definir, entrenar y evaluar el modelo BERT de clasificación.

### transformers (Hugging Face)
Librería que proporciona arquitecturas de modelos pre-entrenados de última generación (BERT, RoBERTa, DistilBERT, etc.). Incluye tokenizadores, configuraciones y métodos para fine-tuning. Es el estándar de la industria para NLP.

### sentence-transformers
Extensión de transformers para generar embeddings (representaciones vectoriales densas) de oraciones. Útil para métricas de similitud semántica y sistemas de recomendación.

---

## Procesamiento de Datos y Machine Learning

### pandas
Biblioteca para manipulación y análisis de datos. Proporciona DataFrames optimizados para manejar datos tabulares (CSV, Excel). Usada para cargar y preprocesar el dataset de currículums.

### numpy
Computación numérica básica. Proporciona arrays multidimensionales y funciones matemáticas de alto rendimiento. Base de pandas y operaciones matriciales.

### scikit-learn
Biblioteca classical de ML. Incluye métricas (accuracy, F1-score), validación cruzada, y preprocesamiento. Usada para split de datos y evaluación.

### faiss-cpu
Biblioteca de Meta para búsqueda de similitud en espacios vectoriales. Implementa índices eficientes (IVF, HNSW) para buscar vectores similares en millones de elementos.

---

## NLP y Procesamiento de Texto

### spacy
Framework industrial de NLP. Proporciona:
- **NER**: Reconocimiento de entidades (personas, organizaciones, fechas)
- **POS tagging**: Etiquetado de categorías gramaticales
- **Lemmatización**: Reducción de palabras a su forma raíz
- **Dependency parsing**: Análisis de dependencias sintácticas

Modelo usado: `en_core_web_sm` (small English model).

### deep-translator
Interfaz unificada para múltiples APIs de traducción (Google Translate, Deepl, etc.). Traduce automáticamente texto del español al inglés antes de procesarlo.

### langdetect
Biblioteca para detección automática de idioma basada en el algoritmo n-gram. Evita traducir textos que ya están en inglés.

---

## Extracción de Texto de PDFs

### pymupdf (fitz)
Binding de Python para MuPDF. Extrae texto de PDFs digitales (con texto embebido). También convierte páginas a imágenes para OCR.

### pytesseract
Wrapper de Python para Tesseract OCR (motor de reconocimiento óptico de caracteres de Google). Convierte imágenes a texto. Necesita Tesseract instalado en el sistema.

### Pillow
Biblioteca de procesamiento de imágenes de Python. Maneja la conversión de páginas PDF a imágenes antes de pasar a Tesseract.

---

## Aplicación Web

### Django
Framework web full-stack de Python. Proporciona:
- Servidor de desarrollo integrado
- ORM para base de datos
- Sistema de URLs y vistas
- Templates HTML
- Manejo de archivos estáticos y media

### python-dotenv
Carga variables de entorno desde archivos `.env`. Separa configuración sensible (API keys, paths) del código fuente.

---

## Utilidades

### tqdm
Biblioteca para mostrar barras de progreso en iterables. Usada durante el entrenamiento del modelo para visualizar progreso por época.

---

## Resumen para el Profesor

El proyecto utiliza un pipeline completo de NLP:

1. **Extracción de texto**: pymupdf (PDFs digitales) + pytesseract (PDFs escaneados)
2. **Preprocesamiento**: langdetect + deep-translator (traducción) + spacy (NER, lemmatización)
3. **Modelo**: torch + transformers (BERT fine-tuned para clasificación de 42 categorías)
4. **Interfaz**: Django (servidor web)
5. **Datos**: pandas + scikit-learn (manejo del dataset)

Cada dependencia cumple un rol específico en este pipeline de extremo a extremo.