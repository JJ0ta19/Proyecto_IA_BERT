# Django y Tecnologías Web del Proyecto

## 1. DJANGO (Framework Web)

### ¿Qué es Django?
Django es un **framework de desarrollo web** escrito en Python. Es como un "esqueleto" completo que proporciona estructura para crear aplicaciones web rápidamente.

### ¿Para qué se usa en este proyecto?
Django maneja toda la **interfaz web** del sistema:
- Servidor web que recibe las peticiones de los usuarios
- Rutas (URLs) que dirigen las peticiones a las vistas correctas
- Plantillas HTML que generan las páginas que ves
- Gestión de archivos subidos (los PDFs que subes)
- Base de datos (sqlite3)

### Componentes de Django usados:

| Componente | Función en el proyecto |
|------------|------------------------|
| `django.shortcuts.render` | Renderiza plantillas HTML con datos |
| `django.core.files.storage` | Guarda archivos subidos (PDFs) |
| `django.conf.settings` | Configuración global |
| `django.urls.path` | Define las rutas/URLs |
| `django.views` | Funciones que procesan las peticiones |

### Analogía para el profesor:
> Django es como el **constructo de un edificio**: proporciona las paredes, puertas, y estructura base. Tú solo necesitas poner los muebles (lógica de negocio) y decoración (plantillas HTML).

---

## 2. PYTHON-DOTENV

### ¿Qué es?
Una librería que permite guardar **configuración sensible** (contraseñas, API keys, rutas) en un archivo `.env` sin subirlo al código.

### ¿Para qué se usa?
En este proyecto se usa para:
- Rutas de archivos
- Configuraciones de base de datos
- Variables de entorno del sistema

### Ejemplo:
```python
# Sin .env: contraseña expuesta en código
API_KEY = "mi_contra123"

# Con python-dotenv: se lee desde archivo .env
API_KEY = os.getenv("API_KEY")
```

---

## 3. PIL (Pillow) - Procesamiento de Imágenes

### ¿Qué es?
Pillow es la librería más popular de Python para **manipular imágenes**.

### ¿Para qué se usa en el proyecto?
Cuando el PDF está escaneado (es una imagen), necesitamos:
1. Convertir cada página del PDF a imagen → `fitz.Matrix(2,2).get_pixmap()`
2. Abrir esa imagen en memoria → `Image.open(io.BytesIO(img_data))`
3. Enviar la imagen a Tesseract para OCR → `pytesseract.image_to_string()`

### Sin Pillow:
No podríamos procesar PDFs escaneados, solo PDFs digitales.

---

## 4. TESSERACT (Motor de OCR)

### ¿Qué es?
Tesseract es un **motor de reconocimiento óptico de caracteres** (OCR) desarrollado por Google. No es Python, es un programa externo que Python llama.

### ¿Para qué se usa?
- **PDFs digitales**: PyMuPDF extrae el texto directamente
- **PDFs escaneados**: Tesseract "lee" la imagen y convierte letras en texto

### Instalación:
Tesseract debe estar instalado en el sistema operativo:
- Windows: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Linux: `sudo apt install tesseract-ocr`

### Analogía:
> Tesseract es como una **persona que transcribe** lo que ve en una imagen. Le das una foto de un documento y te devuelve el texto escrito.

---

## 5. PYTESTERACT (Bridge de Python a Tesseract)

### ¿Qué es?
pytesseract es un **wrapper** (puente) que permite usar Tesseract desde Python de forma sencilla.

### Código típico:
```python
import pytesseract
from PIL import Image

# Configurar path a Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Leer texto de imagen
text = pytesseract.image_to_string(image, lang='eng')
```

---

## 6. FLASK vs DJANGO (Pregunta común)

### ¿Por qué no usar Flask?

| Característica | Flask | Django |
|----------------|-------|--------|
| Tipo | Micro-framework | Full-stack |
| Base de datos | No incluida | ORM incluido |
| Autenticación | Manual | Incluida |
| Estructura | Flexible | Opinionated |
| Tamaño proyecto | Pequeño-mediano | Grande |

**Para este proyecto:**
- Flask podría funcionar, pero Django proporciona más estructura lista
- Django maneja mejor la seguridad de archivos subidos
- Django tiene mejor integración con templates

---

## 7. Resumen Técnico para el Profesor

### Pipeline completo del proyecto:

```
Usuario sube PDF
    ↓
Django recibe petición (views.py)
    ↓
PyMuPDF (PDF digital) o Tesseract (PDF escaneado)
    ↓
GoogleTranslator (traducción al inglés)
    ↓
spaCy (NER, lemmatización, extracción de secciones)
    ↓
TextCleaner (limpieza de texto)
    ↓
BERT tokenizer (convertir texto a números)
    ↓
Modelo BERT (predicción de categoría)
    ↓
Django renderiza resultado en HTML
    ↓
Usuario ve la predicción
```

### ¿Por qué tantas tecnologías?
Cada tecnología tiene un **propósito específico**:

| Tecnología | Propósito |
|------------|-----------|
| Django | Servidor web y estructura |
| PyMuPDF | Leer PDFs digitales |
| Tesseract | Leer PDFs escaneados |
| PIL | Manipular imágenes |
| GoogleTranslator | Traducción automática |
| spaCy | Procesamiento de lenguaje natural |
| BERT | Modelo de machine learning |
| Torch | Framework de deep learning |

---

## 8. Respuestas rápidas para el profesor

**¿Qué es Django?**
R: Es un framework (estructura) de Python para crear aplicaciones web. Maneja rutas, bases de datos, plantillas y seguridad.

**¿Por qué se usa?**
R: Porque proporciona todo lo necesario para hacer un sitio web sin escribir código desde cero. Es como usar bloques de construcción.

**¿Qué hace exactamente en el proyecto?**
R: Recibe el PDF que subes, lo pasa por todo el pipeline de procesamiento, y te muestra el resultado en el navegador.

**¿Qué pasarías sin Django?**
R: Deberías escribir tu propio servidor web desde cero, lo cual es muy complejo y requiere conocimientos de HTTP, sockets, etc.

**¿Qué es Tesseract?**
R: Es un programa de Google que lee texto de imágenes. Lo usamos para PDFs escaneados que no tienen texto digital.

**¿Y PIL?**
R: Es una librería de Python para editar imágenes. La usamos para convertir páginas PDF a imágenes antes de pasarlas a Tesseract.

**¿Por qué tantos pasos para procesar un PDF?**
R: Cada paso tiene un propósito específico:
1. Extraer texto (OCR)
2. Traducir (porque el modelo está entrenado en inglés)
3. Limpiar (eliminar ruido)
4. Analizar (detectar secciones)
5. Clasificar (BERT)