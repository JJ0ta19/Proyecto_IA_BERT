"""
Script de prueba rápido para entrenar el clasificador
"""
import sys
sys.path.insert(0, '.')

from src.datasets.data_loader import DatasetLoader
from src.datasets.preprocessor import DatasetPreprocessor
from src.models.bert_classifier import BertClassifierModel

print("1. Cargando datasets...")
loader = DatasetLoader()
datasets = loader.load_all()
print(f"   Datasets cargados: {list(datasets.keys())}")

print("\n2. Preparando datos...")
df = datasets['resume_class_training']
preprocessor = DatasetPreprocessor()
texts, labels = preprocessor.prepare_classification_data(df, "Resume Text", "Category")
train_texts, val_texts, train_labels, val_labels = preprocessor.prepare_for_bert(texts, labels, val_size=0.2)
label_mapping, reverse_mapping = preprocessor.get_label_mapping()
print(f"   Train: {len(train_texts)}, Val: {len(val_texts)}")
print(f"   Categorías: {len(label_mapping)}")

print("\n3. Entrenando clasificador (1 época para probar)...")
num_classes = len(label_mapping)
classifier = BertClassifierModel(num_classes=num_classes, device='cpu')
classifier.train(
    train_texts[:100], train_labels[:100],  # Solo 100 samples para probar
    val_texts[:20], val_labels[:20],
    epochs=1,
    batch_size=8,
    save_path='models/test_classifier.pt'
)
print("   Entrenamiento completado!")

print("\n4. Predicción de ejemplo...")
test_text = "Python developer with Django and PostgreSQL experience"
prediction = classifier.predict([test_text])
print(f"   Predicción: {reverse_mapping[prediction[0]]}")

print("\n✅ Todo funcionando!")