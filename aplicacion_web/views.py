"""
=================================================================
PROYECTO DE ANÁLISIS DE CURRÍCULUMS CON BERT
Universidad - Sistema de Clasificación de Hojas de Vida

Este módulo contiene las vistas de Django que orquestan todo el 
pipeline de procesamiento: desde la extracción de texto del PDF
hasta la predicción de categoría profesional usando BERT.

=================================================================
"""

# ====================================================================
# IMPORTS - LIBRERÍAS EXTERNAS
# ====================================================================

import os                      # Manipulación de rutas del sistema operativo
import io                      # Operaciones de entrada/salida en memoria
import torch                   # Framework de deep learning (PyTorch)
import fitz                    # PyMuPDF - extracción de texto de PDFs digitales
from django.shortcuts import render, redirect  # Renderizado de plantillas HTML y redirección
from django.http import HttpResponse, HttpResponseRedirect  # Respuestas HTTP
from django.core.files.storage import FileSystemStorage  # Manejo de archivos subidos
from django.conf import settings          # Configuración global de Django
from deep_translator import GoogleTranslator  # API de traducción automática
import sys                          # Manipulación del sistema e imports

# Agregar la carpeta 'red_neuronal' al path de Python para poder importar sus módulos
sys.path.insert(0, os.path.join(settings.BASE_DIR, 'red_neuronal'))

# ====================================================================
# IMPORTS - MÓDULOS PROPIOS DEL PROYECTO
# NOTA: Los imports de modelos se hacen lazy (dentro de funciones) para evitar
#       que PyTorch cargue al inicio y consuma toda la memoria en producción
# ====================================================================

# Estos imports se hacen dentro de las funciones para evitar consumo de memoria al inicio
# from src.models.bert_classifier import BertClassifierModel
# from src.preprocessing.text_cleaner import TextCleaner
# from src.preprocessing.skill_extractor import SkillExtractor
# from src.preprocessing.nlp_processor import get_nlp_processor

import pytesseract                # OCR - reconocimiento óptico de caracteres
from PIL import Image             # Pillow - procesamiento de imágenes

# ====================================================================
# CONFIGURACIÓN DE TESSERACT OCR
# ====================================================================

# Path al ejecutable de Tesseract (motor de OCR)
# IMPORTANTE: Tesseract debe estar instalado en el sistema
# En Windows usar path local, en Linux/Render usar valor por defecto del sistema
import platform
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# ====================================================================
# RUTAS Y CONSTANTES
# ====================================================================

# Ruta al modelo BERT entrenado (archivo .pt con los pesos)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'red_neuronal', 'models', 'bert_classifier_category.pt')

# Ruta al dataset de entrenamiento (CSV con 10,000 currículums categorizados)
DATASET_PATH = os.path.join(settings.BASE_DIR, 'datos_entrenamiento', '1_resume_classification', 'training_data.csv')

# ID del modelo en Google Drive (desde la documentación)
MODEL_GOOGLE_DRIVE_ID = "1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um"


def ensure_model_exists():
    """
    Verifica si el modelo existe y lo descarga si no existe.
    
    Proceso:
    1. Verificar si el archivo del modelo existe en MODEL_PATH
    2. Si no existe, crear la carpeta models si no existe
    3. Descargar el modelo desde Google Drive usando gdown
    4. Si existe, no hace nada
    
    Esta función se llama automáticamente al cargar el clasificador.
    """
    import gdown
    
    # Verificar si el modelo ya existe
    if os.path.exists(MODEL_PATH):
        print(f"✓ Modelo ya existe: {MODEL_PATH}")
        return True
    
    # Crear directorio si no existe
    model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)
    
    # Descargar modelo desde Google Drive
    print("⚠️ Modelo no encontrado. Descargando desde Google Drive...")
    print(f"   Esto puede tomar varios minutos (archivo ~400MB)...")
    
    try:
        url = f'https://drive.google.com/uc?id={MODEL_GOOGLE_DRIVE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
        print(f"✓ Modelo descargado exitosamente: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"✗ Error al descargar el modelo: {e}")
        return False

# ====================================================================
# VARIABLES GLOBALES (Singleton Pattern)
# ====================================================================

# Classifier: instancia del modelo BERT (se carga una sola vez)
classifier = None

# label_mapping: mapeo de categoría -> índice numérico (e.g., {"Technology": 0, "HR": 1})
label_mapping = None

# reverse_mapping: mapeo de índice numérico -> categoría (e.g., {0: "Technology", 1: "HR"})
reverse_mapping = None




# ====================================================================
# FUNCIÓN: load_mappings()
# ====================================================================
def load_mappings():
    """
    Carga el mapeo de categorías del dataset.
    Los imports se hacen aquí (lazy loading) para evitar cargar spacy/transformers al inicio.
    """
    global label_mapping, reverse_mapping
    if label_mapping is None:
        # Imports lazy para evitar cargar librerías pesadas al inicio
        from src.datasets.data_loader import DatasetLoader
        from src.datasets.preprocessor import DatasetPreprocessor

        # DatasetLoader carga el dataset de entrenamiento
        loader = DatasetLoader()
        df = loader.load_training_data()

        # DatasetPreprocessor prepara los datos para BERT
        preprocessor = DatasetPreprocessor()
        texts, labels = preprocessor.prepare_classification_data(df, "Resume Text", "Category")

        # Generar mapeos: categoría <-> índice numérico
        label_mapping, reverse_mapping = preprocessor.get_label_mapping()

    return label_mapping, reverse_mapping


# ====================================================================
# FUNCIÓN: load_classifier()
# ====================================================================
def load_classifier():
    """
    Carga el modelo BERT una única vez (Singleton).
    IMPORTANTE: Los imports se hacen aquí (lazy loading) para evitar que PyTorch
    se cargue al inicio ycause problemas de memoria en producción.
    """
    global classifier
    if classifier is None:
        # Importar aquí (lazy) para evitar cargar PyTorch al inicio
        from src.models.bert_classifier import BertClassifierModel

        # Verificar y descargar modelo si no existe
        if not ensure_model_exists():
            print("✗ ERROR: No se pudo obtener el modelo. La aplicación no funcionará correctamente.")

        # Obtener número de categorías (e.g., 42)
        label_map, reverse_map = load_mappings()
        num_classes = len(label_map)

        # Crear modelo con 42 clases de salida, usando CPU
        classifier = BertClassifierModel(num_classes=num_classes, device='cpu')
        classifier.model_loaded = False

        # Cargar pesos entrenados si existen
        if os.path.exists(MODEL_PATH):
            classifier.load_model(MODEL_PATH)
            classifier.model_loaded = True
            print(f"✓ Modelo entrenado cargado desde: {MODEL_PATH}")
            print(f"  - num_classes: {classifier.num_classes}")
            print(f"  - device: {classifier.device}")
        else:
            print(f"✗ ADVERTENCIA: Modelo no encontrado en: {MODEL_PATH}")
            print(f"  - El modelo usará pesos aleatorios (sin entrenamiento)")

    return classifier


# ====================================================================
# FUNCIÓN: extract_text_from_pdf(pdf_file)
# ====================================================================
def extract_text_from_pdf(pdf_file):
    """
    Extrae texto de un PDF usando OCR.
    
    Pipeline:
    1. Abre el PDF con fitz (PyMuPDF)
    2. Convierte cada página a imagen (para garantizar extracción)
    3. Usa pytesseract (Tesseract OCR) para convertir imagen a texto
    4. Concatena el texto de todas las páginas
    
    Args:
        pdf_file: ruta al archivo PDF
    
    Returns:
        str: Texto extraído del PDF
    
    Dependencias:
        - fitz (PyMuPDF): apertura y renderizado de PDFs
        - PIL (Pillow): conversión de página a imagen
        - pytesseract: OCR para reconocimiento de texto en imágenes
        - Tesseract: motor de OCR (debe estar instalado en sistema)
    """
    text = ""
    try:
        doc = fitz.open(pdf_file)  # Abrir documento PDF
        for page_num, page in enumerate(doc):
            print(f"Procesando página {page_num + 1}...")
            # Convertir página a imagen con escala 2x para mejor calidad
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            # Abrir imagen desde bytes
            image = Image.open(io.BytesIO(img_data))
            # OCR: convertir imagen a texto (idioma inglés)
            ocr_text = pytesseract.image_to_string(image, lang='eng')
            print(f"  Extraídos {len(ocr_text)} caracteres")
            text += ocr_text or ""
        doc.close()
        print(f"Total extraído: {len(text)} caracteres")
    except Exception as e:
        print(f"Error: {e}")
    return text


# ====================================================================
# FUNCIÓN: is_english(text, threshold=0.5)
# ====================================================================
def is_english(text, threshold=0.5):
    """
    Detecta si un texto está en inglés.
    
    Método:
    1. Intenta usar langdetect (detector estadístico de idioma)
    2. Si falla, usa método heurístico basado en palabras clave
    
    Args:
        text: texto a analizar
        threshold: porcentaje mínimo de palabras en inglés para considerar inglés
    
    Returns:
        bool: True si el texto está en inglés
    
    Dependencias:
        - langdetect: biblioteca de detección de idioma
    """
    try:
        from langdetect import detect, LangDetectException
        lang = detect(text[:500])  # Usar primeros 500 caracteres
        return lang == 'en'
    except:
        # Método alternativo: palabras clave en inglés
        words = text.lower().split()
        if not words:
            return True
        # Conjunto de palabras comunes en inglés
        english_words = {'the', 'and', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 
                        'to', 'for', 'of', 'with', 'by', 'a', 'an', 'i', 'you', 'he', 
                        'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 
                        'our', 'their', 'software', 'engineer', 'developer', 
                        'programmer', 'systems', 'technical', 'support', 'experience', 
                        'education', 'skills', 'python', 'java', 'javascript', 
                        'database', 'web', 'network', 'manager', 'analyst', 'designer', 
                        'data', 'project', 'management', 'team', 'working', 
                        'knowledge', 'professional', 'work', 'company'}
        
        # Contar palabras que están en el conjunto de palabras clave
        english_count = sum(1 for w in words if w in english_words)
        return (english_count / len(words)) > threshold


# ====================================================================
# FUNCIÓN: translate_to_english(text)
# ====================================================================
def translate_to_english(text):
    """
    Traduce texto al inglés si no lo está ya.
    
    Pipeline:
    1. Verifica si el texto ya está en inglés (is_english)
    2. Si no, traduce usando GoogleTranslator
    3. Maneja textos largos dividiéndolos en chunks
    
    Args:
        text: texto en español u otro idioma
    
    Returns:
        str: texto traducido al inglés
    
    Dependencias:
        - GoogleTranslator (deep-translator): API de traducción
    """
    if not text or len(text.strip()) < 20:
        return text
    
    # Si ya está en inglés, no traducir
    if is_english(text):
        return text
    
    try:
        translator = GoogleTranslator(source='es', target='en')
        max_length = 4500  # Límite de Google Translator
        
        if len(text) > max_length:
            # Dividir texto en chunks si es muy largo
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            translated_chunks = [translator.translate(chunk) for chunk in chunks]
            return ' '.join(translated_chunks)
        
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


# ====================================================================
# FUNCIÓN: extract_keywords(text, top_n=10)
# ====================================================================
def extract_keywords(text, top_n=10):
    """
    Extrae las palabras clave más frecuentes del texto.
    
    Proceso:
    1. Tokeniza el texto (solo palabras de 3+ letras)
    2. Elimina stopwords (palabras comunes)
    3. Cuenta frecuencia de cada palabra
    4. Retorna las top N más comunes
    
    Args:
        text: texto a analizar
        top_n: número de palabras clave a retornar
    
    Returns:
        list: [(palabra, frecuencia), ...]
    
    Dependencias:
        - re: expresiones regulares para tokenización
        - collections.Counter: contador de frecuencias
    """
    import re
    from collections import Counter
    
    text_lower = text.lower()
    # Extraer palabras: solo letras, mínimo 3 caracteres
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    
    # Stopwords: palabras muy comunes que no aportan información
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 
                 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 
                 'were', 'being', 'with', 'this', 'that', 'from', 'they', 'will', 
                 'what', 'when', 'your', 'more', 'about', 'also', 'into', 'year', 
                 'some', 'could', 'them', 'than', 'then', 'now', 'look', 'only', 
                 'come', 'its', 'over', 'such', 'take', 'just'}
    
    # Filtrar stopwords
    words = [w for w in words if w not in stop_words]
    
    # Retornar las más comunes
    return Counter(words).most_common(top_n)


# ====================================================================
# FUNCIÓN: get_dataset_info()
# ====================================================================
def get_dataset_info():
    """
    Obtiene información del dataset de entrenamiento.
    
    Returns:
        dict: {total_samples: int, category_counts: {categoría: cantidad}}
    
    Dependencias:
        - pandas: lectura del CSV
    """
    import pandas as pd
    try:
        df = pd.read_csv(DATASET_PATH)
        category_counts = df['Category'].value_counts()
        label_map, _ = load_mappings()
        info = {}
        for idx, cat in label_map.items():
            info[cat] = category_counts.get(cat, 0)
        return {'total_samples': len(df), 'category_counts': info}
    except:
        return {'total_samples': 0, 'category_counts': {}}


# ====================================================================
# FUNCIÓN: predict_category(text, classifier)
# ====================================================================
def predict_category(text, classifier):
    """
    Función principal de predicción.
    Los imports se hacen aquí (lazy loading) para evitar cargar PyTorch al inicio.
    """
    # Imports lazy para evitar cargar PyTorch al inicio de la app
    from src.preprocessing.text_cleaner import TextCleaner
    from src.preprocessing.skill_extractor import SkillExtractor
    from src.preprocessing.nlp_processor import get_nlp_processor

    # Cargar mapeos de categorías
    label_map, reverse_map = load_mappings()
    dataset_info = get_dataset_info()

    if not text or len(text.strip()) < 10:
        return None, []

    # Paso 1: Traducir al inglés
    text_en = translate_to_english(text)

    # Paso 2: Procesamiento NLP completo (NER, lemmatización, etc.)
    print("Procesando NLP...")
    nlp = get_nlp_processor()
    nlp_result = nlp.process(text_en)
    sections = nlp.extract_sections(text_en)
    
    # Paso 3: Extraer secciones relevantes del CV
    relevant_sections = ""
    
    if sections.get('experience'):
        relevant_sections += "EXPERIENCE: " + sections['experience'] + " "
    
    if sections.get('education'):
        relevant_sections += "EDUCATION: " + sections['education'] + " "
    
    if sections.get('skills'):
        relevant_sections += "SKILLS: " + sections['skills'] + " "
    
    if sections.get('projects'):
        relevant_sections += "PROJECTS: " + sections['projects'] + " "
    
    # Si no hay secciones, usar texto completo
    if not relevant_sections.strip():
        relevant_sections = text_en
    
    # Paso 4: Extraer entidades y enriquecer texto
    tech_entities = nlp_result['entities'].get('TECH', [])
    skills_from_ner = nlp.get_skills_from_ner(text_en)
    all_skills = list(set(tech_entities + skills_from_ner))
    
    org_entities = nlp_result['entities'].get('ORG', [])[:5]
    person_entities = nlp_result['entities'].get('PERSON', [])[:3]
    
    keywords_text = extract_keywords(text_en, top_n=15)
    keywords_list = [kw[0] for kw in keywords_text] if keywords_text else []
    
    # Agregar entidades detectadas al texto
    relevant_sections += " [TECH: " + ", ".join(all_skills) + "] "
    relevant_sections += " [KEYWORDS: " + ", ".join(keywords_list) + "] "
    relevant_sections += " [COMPANIES: " + ", ".join(org_entities) + "] "
    
    # Paso 5: Limpiar texto (eliminar caracteres especiales, etc.)
    cleaned_text = TextCleaner().clean(relevant_sections)

    # Paso 6: Tokenización y predicción con BERT
    classifier.model.eval()  # Modo evaluación (sin dropout)
    encoding = classifier.tokenizer(
        cleaned_text, 
        add_special_tokens=True,
        max_length=512,          # Longitud máxima de BERT
        padding='max_length',    # Rellenar a 512 tokens
        truncation=True,         # Truncar si es más largo
        return_attention_mask=True,
        return_tensors='pt'      # Retornar tensores de PyTorch
    )

    input_ids = encoding['input_ids'].to(classifier.device)
    attention_mask = encoding['attention_mask'].to(classifier.device)

    # Paso 7: Inferencia (sin calcular gradientes)
    with torch.no_grad():
        outputs = classifier.model(input_ids, attention_mask)
        # Aplicar softmax para obtener probabilidades
        probs = torch.softmax(outputs, dim=1)

    # Paso 8: Extraer resultados
    probs = probs[0].cpu().numpy()
    predicted_class = probs.argmax()  # Índice de la clase con mayor probabilidad
    confidence = probs[predicted_class] * 100  # Porcentaje de confianza

    # Crear lista de todas las predicciones con probabilidades
    results = []
    for idx, prob in enumerate(probs):
        category = reverse_map.get(idx, f"Category_{idx}")
        results.append({'category': category, 'probability': float(prob * 100)})
    results.sort(key=lambda x: x['probability'], reverse=True)

    # Extraer skills usando SkillExtractor
    skills_found = SkillExtractor().extract_skills(text_en)

    # Obtener información del modelo
    total_params = sum(p.numel() for p in classifier.model.parameters())
    bert_config = classifier.model.bert.config

    # Retornar diccionario con todos los resultados
    return {
        'category': reverse_map.get(predicted_class),
        'confidence': float(confidence),
        'nlp_entities': nlp_result['entities'],           # Entidades NER
        'nlp_lemmas': nlp_result['lemmatized'][:100],    # Palabras lematizadas
        'nlp_pos_tags': nlp_result['pos_tags'],          # Etiquetas POS
        'nlp_sections': sections,                       # Secciones extraídas
        'skills_from_ner': all_skills,                  # Skills de NER
        'skills': skills_found,                          # Skills del extractor
        'text_extracted': text,                         # Texto original del PDF
        'text_preprocessed': text_en,                   # Texto traducido
        'text_for_model': cleaned_text,                 # Texto limpio para BERT
        'keywords': keywords_list,                      # Palabras clave
        'model_info': {
            'total_training_samples': dataset_info['total_samples'],
            'num_categories': len(label_map),
            'model_loaded': classifier.model_loaded,
            'parameters': {
                'total_parameters': total_params,        # Total de parámetros (110M)
                'num_hidden_layers': bert_config.num_hidden_layers,  # 12 capas
                'hidden_size': bert_config.hidden_size,  # 768 dimensiones
                'classifier_layers': classifier.num_classes  # 42 clases
            }
        },
        'top_3_predictions': results[:3]
    }, results[:10]


# ====================================================================
# VISTAS DE DJANGO
# ====================================================================

def home(request):
    """Vista de la página principal - muestra resultado si existe en sesión."""
    # Obtener resultado de la sesión si existe
    result = request.session.pop('cv_result', None)
    top_categories = request.session.pop('cv_top_categories', None)
    error = request.session.pop('cv_error', None)
    
    response = render(request, 'analyzer/index.html', 
                      {'result': result, 'top_categories': top_categories, 'error': error})
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response


def upload_cv(request):
    """
    Vista para subir CV desde la página principal.
    
    Proceso:
    1. Recibe archivo PDF via POST
    2. Lo guarda temporalmente
    3. Extrae texto del PDF
    4. Predice categoría
    5. Elimina archivo temporal
    6. Renderiza resultado
    
    Nota: Cada vez que se recarga la página, se debe subir un nuevo CV.
    """
    # Siempre redirigir GET a home (forzar nuevo análisis)
    if request.method == 'GET':
        response = redirect('home')
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        return response
    
    result = None
    top_categories = []
    
    # Verificar que sea POST y tenga archivo
    if request.FILES.get('cv_file'):
        cv_file = request.FILES['cv_file']
        
        # Validar que sea PDF
        if not cv_file.name.lower().endswith('.pdf'):
            request.session['cv_error'] = 'Sube un archivo PDF'
            return redirect('home')
        
        # Guardar temporalmente
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(cv_file.name, cv_file)
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        try:
            # Extraer texto del PDF
            text = extract_text_from_pdf(file_path)
            
            if not text or len(text.strip()) < 10:
                request.session['cv_error'] = 'No se pudo extraer texto del PDF'
                return redirect('home')
            
            # Cargar modelo y predecir
            clf = load_classifier()
            result, top_categories = predict_category(text, clf)
            
            # Guardar resultado en sesión y redirigir (PRG pattern)
            request.session['cv_result'] = result
            request.session['cv_top_categories'] = top_categories
            response = redirect('home')
            response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response['Pragma'] = 'no-cache'
            response['Expires'] = '0'
            return response
            
        finally:
            # Eliminar archivo temporal
            if os.path.exists(file_path):
                os.remove(file_path)
    
    # Si no hay archivo, mostrar formulario vacío
    response = redirect('home')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


def model_info(request):
    """Vista de información del modelo."""
    import pandas as pd
    try:
        df = pd.read_csv(DATASET_PATH)
        total_samples = len(df)
        # Obtener las 15 categorías principales para la gráfica
        category_counts = df['Category'].value_counts().head(15)
        category_labels = list(category_counts.index)
        category_values = list(category_counts.values)
    except:
        total_samples = 10000
        category_labels = ['Technology', 'Data & Analytics', 'Healthcare', 'Marketing & Sales', 
                           'Engineering', 'Finance', 'HR', 'Design', 'Education', 'Operations',
                           'Consulting', 'Legal', 'Research', 'Product', 'Quality']
        category_values = [2511, 568, 488, 463, 435, 380, 320, 280, 250, 200, 180, 150, 120, 100, 80]
    
    label_map, _ = load_mappings()
    clf = load_classifier()
    bert_config = clf.model.bert.config
    total_params = sum(p.numel() for p in clf.model.parameters())
    
    return render(request, 'analyzer/model_info.html', {
        'total_samples': total_samples,
        'train_samples': int(total_samples * 0.8),
        'val_samples': int(total_samples * 0.2),
        'num_categories': len(label_map),
        'categories': list(label_map.keys()),
        'category_labels': category_labels,
        'category_values': category_values,
        'category_distribution': category_counts.to_dict() if 'category_counts' in locals() else {},
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': '2e-5',
        'max_length': 512,
        'model_params': {
            'total_params': total_params,
            'num_layers': bert_config.num_hidden_layers,
            'hidden_size': bert_config.hidden_size,
            'num_heads': bert_config.num_attention_heads,
            'vocab_size': bert_config.vocab_size,
        },
        'classifier_classes': clf.num_classes
    })


def analyze_pdf(request):
    """
    Vista alternativa para análisis de PDF.
    Similar a upload_cv pero usa plantilla diferente.
    """
    result = None
    top_categories = []
    
    if request.method == 'POST' and request.FILES.get('cv_file'):
        cv_file = request.FILES['cv_file']
        
        if not cv_file.name.lower().endswith('.pdf'):
            return render(request, 'analyzer/analyze.html', {'error': 'Sube un archivo PDF'})
        
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(cv_file.name, cv_file)
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        try:
            text = extract_text_from_pdf(file_path)
            if not text or len(text.strip()) < 10:
                return render(request, 'analyzer/analyze.html', {'error': 'No se pudo extraer texto'})
            
            clf = load_classifier()
            result, top_categories = predict_category(text, clf)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return render(request, 'analyzer/analyze.html', 
                  {'result': result, 'top_categories': top_categories})