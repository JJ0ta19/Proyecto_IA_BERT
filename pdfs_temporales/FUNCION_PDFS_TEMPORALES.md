# PDFs Temporales

## Función
Almacena archivos subidos temporalmente por los usuarios durante el análisis de currículums.

## Contenido
- PDFs subidos por los usuarios
- Archivos temporales que se eliminan después del procesamiento

## Proceso Automático
1. Usuario sube PDF → Se guarda en `pdfs_temporales/`
2. Se procesa el PDF
3. Después del análisis, el archivo se elimina automáticamente

## Configuración
En Django (`settings.py`):
```python
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'pdfs_temporales')
```

## Notas
- Esta carpeta puede estar vacía después de usar la aplicación
- No almacenar archivos importantes aquí
- El código ya elimina los archivos automáticamente
- Anteriormente llamada "media"