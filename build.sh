#!/bin/bash
# ===========================================
# Script de construcción para Render
# Descarga modelo BERT y configura dependencias
# ===========================================

set -e

echo "=== INICIANDO BUILD PARA RENDER ==="

# 1. Instalar dependencias del sistema (Tesseract OCR)
echo "1. Instalando Tesseract OCR..."
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-spa wget

# 2. Descargar modelo spaCy
echo "2. Descargando modelo spaCy..."
python -m spacy download en_core_web_sm

# 3. Crear directorio para el modelo
echo "3. Preparando directorio del modelo..."
mkdir -p red_neuronal/models

# 4. Descargar modelo BERT desde Google Drive
echo "4. Descargando modelo BERT..."
# El ID del archivo se extrae de la URL de compartir de Google Drive
# Enlace: https://drive.google.com/file/d/15ym6naog-z68qtaz3jsjbTSwH4HfxpEy/view
# ID: 15ym6naog-z68qtaz3jsjbTSwH4HfxpEy

python -c "
import gdown
import os

url = 'https://drive.google.com/uc?id=1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um'
output = 'red_neuronal/models/bert_classifier_category.pt'

# Descargar archivo grande
gdown.download(url, output, quiet=False)
print('Modelo descargado exitosamente!')
"

# 5. Verificar que el modelo existe
echo "5. Verificando modelo..."
ls -lh red_neuronal/models/

echo "=== BUILD COMPLETADO ==="