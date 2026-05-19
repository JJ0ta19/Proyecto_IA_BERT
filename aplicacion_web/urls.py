"""
=================================================================
URLs DE LA APLICACIÓN DJANGO
Universidad - Rutas del Servidor Web

Este archivo define las URLs disponibles en la aplicación:
- / : Página principal (home) - Subir CV y predecir categoría
- /modelo/ : Información del modelo

Cada URL está asociada a una vista en views.py

=================================================================
"""

from django.urls import path  # Función de Django para definir rutas
from . import views          # Importar todas las vistas

# Lista de rutas disponibles
urlpatterns = [
    path('', views.home, name='home'),           # Página principal
    path('upload/', views.upload_cv, name='upload_cv'),  # Subir CV
    path('modelo/', views.model_info, name='model_info'),  # Info del modelo
]