#!/usr/bin/env bash

set -o errexit

echo "=== INICIANDO BUILD PARA RENDER ==="

# 0. Eliminar entorno virtual viejo (si existe) para forzar Python 3.11
echo "0. Limpiando entorno virtual viejo..."
rm -rf .venv

# 1. Crear entorno virtual con Python 3.11
echo "1. Creando entorno virtual con Python 3.11..."
python3.11 -m venv .venv

# 2. Activar entorno virtual
echo "2. Activando entorno virtual..."
source .venv/bin/activate

# 3. Actualizar pip
echo "3. Actualizando pip..."
pip install --upgrade pip

# 4. Instalar dependencias de Python
echo "4. Instalando dependencias..."
pip install -r requirements.txt

# 5. Descargar modelo spaCy
echo "5. Descargando modelo spaCy..."
python -m spacy download en_core_web_sm

# 6. Crear directorio para el modelo
echo "6. Preparando directorio del modelo..."
mkdir -p red_neuronal/models

# 7. Descargar modelo BERT desde Google Drive
echo "7. Descargando modelo BERT..."

python -c "
import gdown
import os

url = 'https://drive.google.com/uc?id=1dXUht0jrVIA8IqWTLEHPHXd-8kgRc1Um'
output = 'red_neuronal/models/bert_classifier_category.pt'

gdown.download(url, output, quiet=False)
print('Modelo descargado exitosamente!')
"

# 8. Verificar modelo
echo "8. Verificando modelo..."
ls -lh red_neuronal/models/

# 9. Recolectar archivos estáticos
echo "9. Recolectando archivos estáticos..."
python manage.py collectstatic --noinput

# 10. Ejecutar migraciones de base de datos
echo "10. Ejecutando migraciones..."
python manage.py migrate

echo "=== BUILD COMPLETADO ==="