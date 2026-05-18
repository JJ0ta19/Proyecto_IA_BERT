import os
import io
import torch
import fitz
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from deep_translator import GoogleTranslator
import sys
sys.path.insert(0, os.path.join(settings.BASE_DIR, 'nucleo_ia'))
from src.models.bert_classifier import BertClassifierModel
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.skill_extractor import SkillExtractor
from src.preprocessing.nlp_processor import get_nlp_processor
from src.datasets.data_loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MODEL_PATH = os.path.join(settings.BASE_DIR, 'nucleo_ia', 'models', 'bert_classifier_category.pt')
DATASET_PATH = os.path.join(settings.BASE_DIR, 'datos_entrenamiento', '1_resume_classification', 'training_data.csv')

classifier = None
label_mapping = None
reverse_mapping = None


def load_mappings():
    global label_mapping, reverse_mapping
    if label_mapping is None:
        loader = DatasetLoader()
        datasets = loader.load_all()
        df = datasets['resume_class_training']
        preprocessor = DatasetPreprocessor()
        texts, labels = preprocessor.prepare_classification_data(df, "Resume Text", "Category")
        label_mapping, reverse_mapping = preprocessor.get_label_mapping()
    return label_mapping, reverse_mapping


def load_classifier():
    global classifier
    if classifier is None:
        label_map, reverse_map = load_mappings()
        num_classes = len(label_map)
        classifier = BertClassifierModel(num_classes=num_classes, device='cpu')
        if os.path.exists(MODEL_PATH):
            classifier.load_model(MODEL_PATH)
    return classifier


def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        doc = fitz.open(pdf_file)
        for page_num, page in enumerate(doc):
            print(f"Procesando página {page_num + 1}...")
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            ocr_text = pytesseract.image_to_string(image, lang='eng')
            print(f"  Extraídos {len(ocr_text)} caracteres")
            text += ocr_text or ""
        doc.close()
        print(f"Total extraído: {len(text)} caracteres")
    except Exception as e:
        print(f"Error: {e}")
    return text


def is_english(text, threshold=0.5):
    try:
        from langdetect import detect, LangDetectException
        lang = detect(text[:500])
        return lang == 'en'
    except:
        words = text.lower().split()
        if not words:
            return True
        english_words = {'the', 'and', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'software', 'engineer', 'developer', 'programmer', 'systems', 'technical', 'support', 'experience', 'education', 'skills', 'python', 'java', 'javascript', 'database', 'web', 'network', 'manager', 'analyst', 'designer', 'data', 'project', 'management', 'team', 'working', 'knowledge', 'professional', 'work', 'company'}
        english_count = sum(1 for w in words if w in english_words)
        return (english_count / len(words)) > threshold

def translate_to_english(text):
    if not text or len(text.strip()) < 20:
        return text
    
    if is_english(text):
        return text
    
    try:
        translator = GoogleTranslator(source='es', target='en')
        max_length = 4500
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            translated_chunks = [translator.translate(chunk) for chunk in chunks]
            return ' '.join(translated_chunks)
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def extract_keywords(text, top_n=10):
    import re
    from collections import Counter
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'were', 'being', 'with', 'this', 'that', 'from', 'they', 'will', 'what', 'when', 'your', 'more', 'about', 'also', 'into', 'year', 'some', 'could', 'them', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'such', 'take', 'just'}
    words = [w for w in words if w not in stop_words]
    return Counter(words).most_common(top_n)


def get_dataset_info():
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


def predict_category(text, classifier):
    label_map, reverse_map = load_mappings()
    dataset_info = get_dataset_info()
    
    if not text or len(text.strip()) < 10:
        return None, []

    text_en = translate_to_english(text)
    
    print("Procesando NLP...")
    nlp = get_nlp_processor()
    nlp_result = nlp.process(text_en)
    sections = nlp.extract_sections(text_en)
    
    relevant_sections = ""
    
    if sections.get('experience'):
        relevant_sections += "EXPERIENCE: " + sections['experience'] + " "
    
    if sections.get('education'):
        relevant_sections += "EDUCATION: " + sections['education'] + " "
    
    if sections.get('skills'):
        relevant_sections += "SKILLS: " + sections['skills'] + " "
    
    if sections.get('projects'):
        relevant_sections += "PROJECTS: " + sections['projects'] + " "
    
    if not relevant_sections.strip():
        relevant_sections = text_en
    
    tech_entities = nlp_result['entities'].get('TECH', [])
    skills_from_ner = nlp.get_skills_from_ner(text_en)
    all_skills = list(set(tech_entities + skills_from_ner))
    
    org_entities = nlp_result['entities'].get('ORG', [])[:5]
    person_entities = nlp_result['entities'].get('PERSON', [])[:3]
    
    keywords_text = extract_keywords(text_en, top_n=15)
    keywords_list = [kw[0] for kw in keywords_text] if keywords_text else []
    
    relevant_sections += " [TECH: " + ", ".join(all_skills) + "] "
    relevant_sections += " [KEYWORDS: " + ", ".join(keywords_list) + "] "
    relevant_sections += " [COMPANIES: " + ", ".join(org_entities) + "] "
    
    cleaned_text = TextCleaner().clean(relevant_sections)

    classifier.model.eval()
    encoding = classifier.tokenizer(cleaned_text, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')

    input_ids = encoding['input_ids'].to(classifier.device)
    attention_mask = encoding['attention_mask'].to(classifier.device)

    with torch.no_grad():
        outputs = classifier.model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)

    probs = probs[0].cpu().numpy()
    predicted_class = probs.argmax()
    confidence = probs[predicted_class] * 100

    results = []
    for idx, prob in enumerate(probs):
        category = reverse_map.get(idx, f"Category_{idx}")
        results.append({'category': category, 'probability': float(prob * 100)})
    results.sort(key=lambda x: x['probability'], reverse=True)

    skills_found = SkillExtractor().extract_skills(text_en)

    total_params = sum(p.numel() for p in classifier.model.parameters())
    bert_config = classifier.model.bert.config

    return {
        'category': reverse_map.get(predicted_class),
        'confidence': float(confidence),
        'nlp_entities': nlp_result['entities'],
        'nlp_lemmas': nlp_result['lemmatized'][:100],
        'nlp_pos_tags': nlp_result['pos_tags'],
        'nlp_sections': sections,
        'skills_from_ner': all_skills,
        'skills': skills_found,
        'text_extracted': text,
        'text_preprocessed': text_en,
        'text_for_model': cleaned_text,
        'keywords': keywords_list,
        'model_info': {
            'total_training_samples': dataset_info['total_samples'],
            'num_categories': len(label_map),
            'parameters': {
                'total_parameters': total_params,
                'num_hidden_layers': bert_config.num_hidden_layers,
                'hidden_size': bert_config.hidden_size,
                'classifier_layers': classifier.num_classes
            }
        },
        'top_3_predictions': results[:3]
    }, results[:10]


def home(request):
    return render(request, 'analyzer/index.html')


def upload_cv(request):
    result = None
    top_categories = []
    if request.method == 'POST' and request.FILES.get('cv_file'):
        cv_file = request.FILES['cv_file']
        if not cv_file.name.lower().endswith('.pdf'):
            return render(request, 'analyzer/index.html', {'error': 'Sube un archivo PDF'})
        
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(cv_file.name, cv_file)
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        try:
            text = extract_text_from_pdf(file_path)
            if not text or len(text.strip()) < 10:
                return render(request, 'analyzer/index.html', {'error': 'No se pudo extraer texto del PDF'})
            
            clf = load_classifier()
            result, top_categories = predict_category(text, clf)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return render(request, 'analyzer/index.html', {'result': result, 'top_categories': top_categories})


def model_info(request):
    import pandas as pd
    try:
        df = pd.read_csv(DATASET_PATH)
        total_samples = len(df)
        category_counts = df['Category'].value_counts().head(10).to_dict()
    except:
        total_samples = 10000
        category_counts = {}
    
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
        'category_distribution': category_counts,
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
    
    return render(request, 'analyzer/analyze.html', {'result': result, 'top_categories': top_categories})