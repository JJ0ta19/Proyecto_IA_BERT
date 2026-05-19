#!/usr/bin/env bash

set -o errexit

echo "=== INICIANDO BUILD PARA RENDER ==="

# 1. Instalar dependencias de Python
echo "1. Instalando dependencias..."
pip install -r requirements.txt

# 2. Instalar dependencias del sistema (Tesseract OCR)
echo "2. Instalando Tesseract OCR..."
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-spa wget

# 3. Descargar modelo spaCy
echo "3. Descargando modelo spaCy..."
python -m spacy download en_core_web_sm

# 4. Crear directorio para el modelo
echo "4. Preparando directorio del modelo..."
mkdir -p red_neuronal/models

# 5. Descargar modelo BERT desde Google Drive
echo "5. Descargando modelo BERT..."

python -c "
import gdown
import os

url = 'https://drive.google.com/uc?id=1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um'
output = 'red_neuronal/models/bert_classifier_category.pt'

gdown.download(url, output, quiet=False)
print('Modelo descargado exitosamente!')
"

# 6. Verificar modelo
echo "6. Verificando modelo..."
ls -lh red_neuronal/models/

# 7. Recolectar archivos estáticos
echo "7. Recolectando archivos estáticos..."
python manage.py collectstatic --noinput

# 8. Ejecutar migraciones de base de datos
echo "8. Ejecutando migraciones..."
python manage.py migrate

echo "=== BUILD COMPLETADO ==="