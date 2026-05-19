#!/usr/bin/env bash

set -o errexit

echo "=== INICIANDO BUILD PARA RENDER ==="

echo "1. Actualizando pip..."
pip install --upgrade pip

echo "2. Instalando torch CPU-only..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "3. Instalando tokenizers pre-built wheel..."
pip install tokenizers>=0.13.0

echo "4. Instalando transformers..."
pip install 'transformers>=4.30.0,<4.40.0'

echo "5. Instalando resto de dependencias..."
pip install -r requirements.txt

echo "6. Descargando modelo spaCy..."
python -m spacy download en_core_web_sm || echo "spaCy download skipped"

echo "7. Preparando directorio del modelo..."
mkdir -p red_neuronal/models

echo "8. Descargando modelo BERT..."
python -c "
import gdown
url = 'https://drive.google.com/uc?id=1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um'
output = 'red_neuronal/models/bert_classifier_category.pt'
gdown.download(url, output, quiet=False)
print('Modelo descargado!')
" || echo "gdown skipped"

echo "9. Recolectando archivos estaticos..."
python manage.py collectstatic --noinput || true

echo "10. Ejecutando migraciones..."
python manage.py migrate || true

echo "=== BUILD COMPLETADO ==="