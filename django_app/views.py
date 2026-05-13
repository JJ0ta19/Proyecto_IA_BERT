import os
import torch
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from deep_translator import GoogleTranslator
import sys
sys.path.insert(0, os.path.join(settings.BASE_DIR, 'neural_network'))
from src.models.bert_classifier import BertClassifierModel
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.skill_extractor import SkillExtractor
from src.datasets.data_loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor

MODEL_PATH = os.path.join(settings.BASE_DIR, 'neural_network', 'models', 'bert_classifier.pt')
DATASET_PATH = os.path.join(settings.BASE_DIR, 'data', '1_resume_classification', 'training_data.csv')

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
        texts, labels = preprocessor.prepare_classification_data(df, "Resume Text", "Job Role")
        label_mapping, reverse_mapping = preprocessor.get_label_mapping()
    return label_mapping, reverse_mapping


def load_classifier():
    global classifier
    if classifier is None:
        label_map, reverse_map = load_mappings()
        num_classes = len(label_map)
        classifier = BertClassifierModel(num_classes=num_classes)
        if os.path.exists(MODEL_PATH):
            classifier.load_model(MODEL_PATH)
    return classifier


def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error con pdfplumber: {e}")
        text = ""
    return text


def translate_to_english(text):
    try:
        return GoogleTranslator(source='es', target='en').translate(text)
    except:
        return text


def extract_keywords(text, top_n=10):
    import re
    from collections import Counter
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'were', 'being', 'with', 'this', 'that', 'from', 'they', 'will', 'what', 'when', 'your', 'more', 'about', 'also', 'into', 'year', 'some', 'could', 'them', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'such', 'take', 'just'}
    words = [w for w in words if w not in stop_words]
    return Counter(words).most_common(top_n)


def extract_education(text):
    import re
    text_lower = text.lower()
    patterns = [
        r"bachelor'?s?\s+in\s+([a-z\s]+)",
        r"master'?s?\s+in\s+([a-z\s]+)",
        r"phd\s+in\s+([a-z\s]+)",
        r"doctorate\s+in\s+([a-z\s]+)",
        r"associate\s+degree",
        r"mba",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(0).upper()
    return "Not specified"


def extract_experience_years(text):
    import re
    text_lower = text.lower()
    patterns = [
        r"(\d+)\s+years?\s+of\s+experience",
        r"experience:\s*(\d+)\s+years?",
        r"(\d+)\+?\s+years?\s+experience",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return int(match.group(1))
    return 0


def get_category_from_job_role(job_role):
    import pandas as pd
    try:
        df = pd.read_csv(DATASET_PATH)
        mapping = df[['Job Role', 'Category']].drop_duplicates()
        category = mapping[mapping['Job Role'] == job_role]['Category'].values
        if len(category) > 0:
            return category[0]
    except:
        pass
    return "Unknown"


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
    cleaned_text = TextCleaner().clean(text_en)

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
        job_role = reverse_map.get(idx, f"Job Role_{idx}")
        results.append({'job_role': job_role, 'probability': float(prob * 100)})
    results.sort(key=lambda x: x['probability'], reverse=True)

    # Extraer Skills
    skills_found = SkillExtractor().extract_skills(text_en)
    keywords = extract_keywords(cleaned_text)
    
    # Extraer Education
    education = extract_education(text_en)
    
    # Extraer Experience Years
    experience_years = extract_experience_years(text_en)
    
    # Obtener Category desde Job Role
    predicted_job_role = reverse_map.get(predicted_class, "Unknown")
    predicted_category = get_category_from_job_role(predicted_job_role)

    total_params = sum(p.numel() for p in classifier.model.parameters())
    bert_config = classifier.model.bert.config

    return {
        'job_role': predicted_job_role,
        'category': predicted_category,
        'education': education,
        'experience_years': experience_years,
        'confidence': float(confidence),
        'skills': skills_found,
        'text_translated': text_en,
        'keywords': keywords,
        'model_info': {
            'total_training_samples': dataset_info['total_samples'],
            'num_job_roles': len(label_map),
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
            if not text or len(text.strip()) < 50:
                return render(request, 'analyzer/index.html', {'error': 'No se pudo extraer texto del PDF'})
            
            clf = load_classifier()
            result, top_categories = predict_category(text, clf)
            result['extracted_text'] = result.get('text_translated', '')[:500] + "..." if len(result.get('text_translated', '')) > 500 else result.get('text_translated', '')
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
            if not text or len(text.strip()) < 50:
                return render(request, 'analyzer/analyze.html', {'error': 'No se pudo extraer texto'})
            
            clf = load_classifier()
            result, top_categories = predict_category(text, clf)
            result['extracted_text'] = result.get('text_translated', '')[:800] + "..." if len(result.get('text_translated', '')) > 800 else result.get('text_translated', '')
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return render(request, 'analyzer/analyze.html', {'result': result, 'top_categories': top_categories})