# Aplicación Web Django

Interfaz web del sistema: permite subir currículums en PDF y obtener la categoría profesional predicha por el modelo BERT.

---

## Rutas disponibles

| Ruta | Vista | Descripción |
|------|-------|-------------|
| `/` | `home` | Página principal con formulario de subida |
| `/upload/` | `upload_cv` | Procesa el PDF subido y muestra la predicción |
| `/modelo/` | `model_info` | Información del modelo con gráficas interactivas |
| `/admin/` | Django admin | Panel de administración |

---

## Vistas principales (`aplicacion_web/views.py`)

| Función | Función |
|---------|---------|
| `home()` | Página principal |
| `upload_cv()` | Procesa el PDF desde la página principal |
| `load_classifier()` | Carga el modelo BERT (con auto-descarga vía gdown si falta) |
| `predict_category()` | Realiza la predicción con probabilidades por categoría |
| `extract_text_from_pdf()` | Extrae texto del PDF (PyMuPDF + OCR Tesseract) |
| `translate_to_english()` | Traduce el texto al inglés |

## Templates (`aplicacion_web/templates/analyzer/`)

| Archivo | Contenido |
|---------|-----------|
| `base.html` | Plantilla base con estilos |
| `index.html` | Página principal con formulario |
| `analyze.html` | Resultado del análisis |
| `model_info.html` | Info del modelo con gráficas Chart.js |

---

## Flujo completo

1. Usuario sube PDF → se guarda en `pdfs_temporales/`
2. Se extrae el texto (PyMuPDF para PDFs digitales, Tesseract OCR para escaneados)
3. Se preprocesa: detección de idioma → traducción al inglés → limpieza NLP
4. El modelo BERT predice la categoría con nivel de confianza
5. Se renderiza el resultado en la vista
6. **El PDF temporal se elimina automáticamente**

---

## Configuración de archivos subidos

En `proyecto_cv/settings.py`:

```python
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'pdfs_temporales'
```

Notas sobre `pdfs_temporales/`:
- Puede quedar vacía después de usar la aplicación (borrado automático)
- No almacenar archivos importantes ahí

---

## Requerimientos

- Django 4.2+
- Modelo BERT entrenado en `red_neuronal/models/bert_classifier_category.pt`
  (auto-descargado desde Google Drive si no existe — ver [01-instalacion.md](01-instalacion.md))
