# Despliegue en Render

Guía de despliegue de la aplicación en [Render](https://render.com) como servicio web con Gunicorn + WhiteNoise.

---

## Archivos de configuración

| Archivo | Propósito |
|---------|-----------|
| `render.yaml` | Definición del servicio (Blueprint de Render) |
| `build.sh` | Script de build: dependencias, modelo y migraciones |
| `runtime.txt` | Versión de Python (`python-3.11.0`) |

---

## Configuración del servicio (`render.yaml`)

```yaml
services:
  - type: web
    name: cv-bert-analyzer
    env: python3
    buildCommand: chmod +x build.sh && ./build.sh
    startCommand: gunicorn proyecto_cv.wsgi:application --bind 0.0.0.0:${PORT:-10000}
    envVars:
      - key: DJANGO_SETTINGS_MODULE
        value: proyecto_cv.settings
```

---

## Proceso de build (`build.sh`)

1. Actualizar pip
2. Instalar **torch CPU-only** (índice oficial de PyTorch; más ligero para producción)
3. Instalar **tokenizers/transformers solo binarios** (`--only-binary :all:` evita compilar Rust)
4. Instalar el resto de `requirements.txt`
5. Descargar modelo spaCy `en_core_web_sm`
6. Crear `red_neuronal/models/`
7. **Descargar el modelo BERT** desde Google Drive con `gdown`
8. `collectstatic` (archivos estáticos con WhiteNoise)
9. Migraciones

---

## Ajustes de Django para producción

En `proyecto_cv/settings.py`:

```python
DEBUG = 'RENDER' not in os.environ          # DEBUG off en Render
ALLOWED_HOSTS = ['*', RENDER_EXTERNAL_HOSTNAME]

# WhiteNoise para archivos estáticos (sin nginx)
MIDDLEWARE = [..., 'whitenoise.middleware.WhiteNoiseMiddleware', ...]
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
```

---

## Optimizaciones aplicadas

- **Lazy loading**: imports pesados (torch, transformers) dentro de funciones para no cargar PyTorch al inicio del servicio
- **torch CPU-only**: imagen más pequeña y compatible
- **tokenizers binarios**: evita fallos de compilación durante el build

---

## Despliegue manual

1. Conectar el repositorio en el dashboard de Render
2. Render detecta `render.yaml` (Blueprint) o configurar manualmente:
   - **Build command:** `chmod +x build.sh && ./build.sh`
   - **Start command:** `gunicorn proyecto_cv.wsgi:application --bind 0.0.0.0:$PORT`
3. Variables de entorno: `DJANGO_SETTINGS_MODULE=proyecto_cv.settings`

---

## Solución de problemas históricos (commits de la rama deploy)

| Problema | Solución aplicada |
|----------|-------------------|
| Python 3.14 incompatible con torch | Forzar Python 3.11 vía `runtime.txt` |
| Compilación de tokenizers falla | `pip install --only-binary :all: tokenizers transformers` |
| Torch demasiado pesado | Instalar desde índice CPU-only |
| Puerto incorrecto en Render | Bind a `0.0.0.0:$PORT` con Gunicorn |
| Carga lenta al inicio | Lazy loading de imports de ML |
