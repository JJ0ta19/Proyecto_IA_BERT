"""
WSGI config for proyecto_cv project.
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'proyecto_cv.settings')

application = get_wsgi_application()