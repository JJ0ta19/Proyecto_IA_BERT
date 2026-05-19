#!/usr/bin/env bash

set -o errexit

echo "=== INICIANDO BUILD PARA RENDER ==="

# 1. Instalar dependencias de Python
echo "1. Instalando dependencias..."
pip install -r requirements.txt

# 2. Descargar modelo spaCy
echo "2. Descargando modelo spaCy..."
python -m spacy download en_core_web_sm

# 3. Crear directorio para el modelo
echo "3. Preparando directorio del modelo..."
mkdir -p red_neuronal/models

# 4. Descargar modelo BERT desde Google Drive
echo "4. Descargando modelo BERT..."

python -c "
import gdown
import os

url = 'https://drive.google.com/uc?id=1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um'
output = 'red_neuronal/models/bert_classifier_category.pt'

gdown.download(url, output, quiet=False)
print('Modelo descargado exitosamente!')
"

# 5. Verificar modelo
echo "5. Verificando modelo..."
ls -lh red_neuronal/models/

# 6. Recolectar archivos estáticos
echo "6. Recolectando archivos estáticos..."
python manage.py collectstatic --noinput

# 7. Ejecutar migraciones de base de datos
echo "7. Ejecutando migraciones..."
python manage.py migrate

echo "=== BUILD COMPLETADO ==="