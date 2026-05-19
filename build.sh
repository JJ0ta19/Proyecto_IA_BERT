#!/usr/bin/env bash

set -o errexit

echo "=== INICIANDO BUILD PARA RENDER ==="

# 0. Limpiar cache de pip para evitar conflictos
echo "0. Limpiando cache de pip..."
rm -rf /opt/render/project/src/.venv

# 1. Actualizar pip (con --break-system-packages para entornos externos)
echo "1. Actualizando pip..."
pip install --upgrade pip --break-system-packages

# 2. Instalar torch primero (sin CUDA para evitar problemas)
echo "2. Instalando torch CPU-only..."
pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# 3. Instalar transformers (versión específica compatible)
echo "3. Instalando transformers..."
pip install 'transformers>=4.30.0,<4.40.0' --break-system-packages

# 4. Instalar el resto de dependencias
echo "4. Instalando dependencias..."
pip install -r requirements.txt --break-system-packages

# 5. Descargar modelo spaCy
echo "5. Descargando modelo spaCy..."
python -m spacy download en_core_web_sm --break-system-packages

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