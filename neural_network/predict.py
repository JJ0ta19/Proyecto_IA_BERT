"""
Predicción de categoría de CV usando modelo entrenado
Si el modelo no existe, entrena automáticamente
"""
import sys
sys.path.insert(0, '.')

import os
from src.data_processing.data_loader import DatasetLoader
from src.data_processing.preprocessor import DatasetPreprocessor
from src.ml_models.bert_classifier import BertClassifierModel

MODEL_PATH = 'models/bert_classifier.pt'

def load_or_train():
    # Cargar datasets
    loader = DatasetLoader()
    datasets = loader.load_all()
    df = datasets['resume_class_training']

    # Preprocesar
    preprocessor = DatasetPreprocessor()
    texts, labels = preprocessor.prepare_classification_data(df, "Resume Text", "Category")
    train_texts, val_texts, train_labels, val_labels = preprocessor.prepare_for_bert(texts, labels, val_size=0.2)
    label_mapping, reverse_mapping = preprocessor.get_label_mapping()
    num_classes = len(label_mapping)

    # Verificar si existe modelo guardado
    if os.path.exists(MODEL_PATH):
        print("✅ Cargando modelo guardado...")
        classifier = BertClassifierModel(num_classes=num_classes, device='cpu')
        classifier.load_model(MODEL_PATH)
    else:
        print("⚠️ No hay modelo guardado. Entrenando...")
        classifier = BertClassifierModel(num_classes=num_classes, device='cpu')
        classifier.train(
            train_texts, train_labels,
            val_texts, val_labels,
            epochs=3,
            batch_size=16,
            save_path=MODEL_PATH
        )
        print("✅ Modelo guardado!")

    return classifier, reverse_mapping

def predict(cv_text):
    classifier, reverse_mapping = load_or_train()

    # Limpiar texto
    from src.text_preprocessing.text_cleaner import TextCleaner
    cleaner = TextCleaner()
    cleaned = cleaner.clean(cv_text)

    # Predecir con probabilidades
    probs = classifier.predict_proba([cleaned])[0]

    # Mostrar resultados
    results = []
    for idx, prob in enumerate(probs):
        cat = reverse_mapping.get(idx, f"Unknown_{idx}")
        results.append({'category': cat, 'probability': prob * 100})

    results.sort(key=lambda x: x['probability'], reverse=True)

    print("\n" + "=" * 60)
    print("PREDICCIÓN DE CATEGORÍA PROFESIONAL")
    print("=" * 60)
    print(f"\n🎯 Categoría: {results[0]['category']}")
    print(f"📊 Confianza: {results[0]['probability']:.2f}%\n")

    print("TOP 5 CATEGORÍAS:")
    print("-" * 40)
    for i, r in enumerate(results[:5]):
        print(f"{i+1}. {r['category']:35} {r['probability']:.2f}%")

# Ejemplo de uso
if __name__ == "__main__":
    cv_example = """
    EXPERIENCE
    Software Engineer at Tech Corp
    - Developed Python applications using Django and Flask
    - Worked with PostgreSQL, MongoDB, and Redis
    - Deployed applications using AWS and Docker

    SKILLS
    Python, JavaScript, Django, AWS, Docker, PostgreSQL

    EDUCATION
    BS Computer Science
    """

    predict(cv_example)