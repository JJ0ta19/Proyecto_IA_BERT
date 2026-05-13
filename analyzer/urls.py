from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_cv, name='upload_cv'),
    path('modelo/', views.model_info, name='model_info'),
    path('analizar/', views.analyze_pdf, name='analyze_pdf'),
]