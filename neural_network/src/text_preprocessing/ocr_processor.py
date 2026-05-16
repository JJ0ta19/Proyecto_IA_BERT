import os
import subprocess
import tempfile
from typing import Optional

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class OCRProcessor:
    def __init__(self, tesseract_path: Optional[str] = None):
        if not TESSERACT_AVAILABLE:
            raise ImportError("Instala pytesseract y Tesseract: pip install pytesseract pillow")

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self._verify_tesseract()

    def _verify_tesseract(self):
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Tesseract no encontrado. Instálalo desde: https://github.com/UB-Mannheim/tesseract/wiki")

    def extract_from_image(self, image_path: str, lang: str = 'spa+eng') -> str:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()

    def extract_from_pdf(self, pdf_path: str, dpi: int = 300, lang: str = 'spa+eng') -> str:
        import pdf2image

        images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        all_text = []

        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang=lang)
            all_text.append(text)

        return '\n'.join(all_text)

    def extract_from_pdf_bytes(self, pdf_bytes: bytes, dpi: int = 300, lang: str = 'spa+eng') -> str:
        import pdf2image
        from PIL import Image
        import io

        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi)
        all_text = []

        for image in images:
            text = pytesseract.image_to_string(image, lang=lang)
            all_text.append(text)

        return '\n'.join(all_text)