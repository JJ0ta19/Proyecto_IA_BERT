from django.apps import AppConfig
import os


class AplicacionWebConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'aplicacion_web'

    def ready(self):
        """Se ejecuta cuando Django inicia. Verifica que el modelo exista."""
        from django.conf import settings
        
        # Ruta del modelo
        model_path = os.path.join(settings.BASE_DIR, 'red_neuronal', 'models', 'bert_classifier_category.pt')
        model_dir = os.path.dirname(model_path)
        
        print("\n" + "="*50)
        print("INICIANDO APLICACIÓN CV ANALYZER")
        print("="*50)
        
        # Verificar si el modelo existe
        if os.path.exists(model_path):
            print(f"✓ Modelo encontrado: {model_path}")
            print("✓ Listo para analizar CVs")
        else:
            print(f"⚠️ Modelo NO encontrado en: {model_path}")
            print("⚠️ Se descargará automáticamente al analizar el primer CV")
            
            # Crear directorio si no existe
            os.makedirs(model_dir, exist_ok=True)
        
        print("="*50 + "\n")