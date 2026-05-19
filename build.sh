#!/usr/bin/env bash

set -o errexit

echo "=== INICIANDO BUILD PARA RENDER ==="

echo "1. Actualizando pip..."
pip install --upgrade pip

echo "2. Instalando torch CPU-only..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "3. Forzando instalacion binaria de tokenizers y transformers..."
pip install --only-binary :all: tokenizers transformers

echo "4. Instalando resto de dependencias..."
pip install -r requirements.txt

echo "5. Descargando modelo spaCy..."
python -m spacy download en_core_web_sm || echo "spaCy download skipped"

echo "6. Preparando directorio del modelo..."
mkdir -p red_neuronal/models

echo "7. Descargando modelo BERT..."
python -c "
import gdown
url = 'https://drive.google.com/uc?id=1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um'
output = 'red_neuronal/models/bert_classifier_category.pt'
gdown.download(url, output, quiet=False)
print('Modelo descargado!')
" || echo "gdown skipped"

echo "8. Recolectando archivos estaticos..."
python manage.py collectstatic --noinput || true

echo "9. Ejecutando migraciones..."
python manage.py migrate || true

echo "=== BUILD COMPLETADO ==="