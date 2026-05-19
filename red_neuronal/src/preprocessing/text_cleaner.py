"""
=================================================================
LIMPIADOR DE TEXTO PARA CURRÍCULUMS
Universidad - Preprocesamiento de Texto

Este módulo contiene funciones para limpiar y normalizar texto
de currículums antes de pasarlo al modelo BERT.

Pipeline de limpieza:
1. Normalizar Unicode (eliminar acentos)
2. Expandir contracciones (don't -> do not)
3. Eliminar caracteres especiales
4. Normalizar espacios en blanco
5. Convertir a minúsculas

=================================================================
"""

import re                      # Expresiones regulares
import string                 # Operaciones con strings
import unicodedata             # Normalización Unicode
from typing import Optional    # Type hints


class TextCleaner:
    """
    Clase para limpiar y normalizar texto de currículums.
    
    Métodos principales:
    - clean(): método principal que aplica todo el pipeline
    - remove_stopwords(): elimina palabras comunes
    - extract_sentences(): divide en oraciones
    - get_tokens(): tokenización básica
    - clean_resume(): limpia texto de CV líneas por línea
    
    Atributos:
    - contractions: dict de contracciones en inglés
    - stopwords: set de palabras comunes a eliminar
    """
    
    def __init__(self):
        """
        Inicializa el limpiador con contracciones y stopwords.
        
        Contracciones: palabras como "don't" -> "do not"
        Stopwords: palabras comunes sin significado semántico
        """
        # Diccionario de contracciones en inglés
        # key: forma contraída, value: forma expandida
        self.contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
            "shouldn't": "should not", "can't": "cannot", "i'm": "i am",
            "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are",
            "i've": "i have", "you've": "you have", "we've": "we have",
            "they've": "they have", "i'll": "i will", "you'll": "you will",
            "he'll": "he will", "she'll": "she will", "we'll": "we will",
            "they'll": "they will", "isn't": "is not", "aren't": "are not",
            "wasn't": "was not", "weren't": "were not", "haven't": "have not",
            "hasn't": "has not", "hadn't": "had not"
        }
        
        # Stopwords: palabras muy comunes en inglés
        # Ejemplo: 'the', 'is', 'are', 'and', 'to', etc.
        # Estas palabras no aportan información semántica
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
            'with', 'about', 'against', 'between', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll',
            'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn',
            'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn',
            'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
        }

    def clean(self, text: str) -> str:
        """
        Pipeline principal de limpieza de texto.
        
        Proceso (en orden):
        1. Validar que sea string no vacío
        2. Normalizar Unicode (eliminar acentos)
        3. Expandir contracciones
        4. Eliminar caracteres especiales y números
        5. Normalizar espacios
        6. Convertir a minúsculas
        
        Args:
            text: texto sucio del currículum
        
        Returns:
            str: texto limpio y normalizado
        
        Ejemplo:
            Input: "I'm a Software Engineer! 😊"
            Output: "i am a software engineer"
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Paso 1: Normalizar caracteres Unicode
        # Elimina acentos: á -> a, é -> e, etc.
        text = self._normalize_unicode(text)
        
        # Paso 2: Expandir contracciones
        # "don't" -> "do not", "I'm" -> "I am"
        text = self._expand_contractions(text)
        
        # Paso 3: Eliminar caracteres especiales
        # Solo deja letras, números y espacios
        # Elimina emoji, símbolos, puntuación
        text = self._remove_special_chars(text)
        
        # Paso 4: Normalizar espacios
        # Múltiples espacios -> uno solo, eliminar bordes
        text = self._normalize_whitespace(text)
        
        # Paso 5: Convertir a minúsculas
        text = text.lower()
        
        return text

    def _normalize_unicode(self, text: str) -> str:
        """
        Normaliza caracteres Unicode (elimina acentos).
        
        Proceso:
        1. Normalizar a forma NFKD (descomposición canónica)
        2. Codificar a ASCII (elimina caracteres no-ASCII)
        3. Decodificar de vuelta a string
        
        Args:
            text: texto con acentos
        
        Returns:
            str: texto sin acentos
        
        Dependencias:
            - unicodedata.normalize: estandarización Unicode
            - encode/decode: conversión ASCII
        """
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    def _expand_contractions(self, text: str) -> str:
        """
        Expande contracciones en inglés.
        
        Proceso:
        1. Recorrer todas las contracciones conocidas
        2. Reemplazar usando regex (respetar word boundaries)
        
        Args:
            text: texto con contracciones
        
        Returns:
            str: texto con contracciones expandidas
        
        Dependencias:
            - re.sub: búsqueda y reemplazo con regex
            - re.IGNORECASE: ignorar mayúsculas/minúsculas
        """
        for key, value in self.contractions.items():
            # \b: word boundary (evita parcial matches)
            # flags=re.IGNORECASE: mayúsculas o minúsculas
            text = re.sub(r'\b' + key + r'\b', value, text, flags=re.IGNORECASE)
        return text

    def _remove_special_chars(self, text: str) -> str:
        """
        Elimina caracteres especiales y números.
        
        Proceso:
        1. Reemplazar todo que no sea letra/espacio con espacio
        2. Eliminar todos los números
        
        Args:
            text: texto con caracteres especiales
        
        Returns:
            str: solo letras y espacios
        
        Dependencias:
            - re.sub: expresiones regulares para limpieza
        """
        # Primero: eliminar todo excepto letras y espacios
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Segundo: eliminar todos los números
        text = re.sub(r'\d+', ' ', text)
        
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normaliza espacios en blanco.
        
        Proceso:
        1. Reemplazar múltiples espacios con uno solo
        2. Eliminar espacios al inicio y final
        
        Args:
            text: texto con espacios irregulares
        
        Returns:
            str: texto con espacios normalizados
        """
        # \s+: uno o más espacios en blanco
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def remove_stopwords(self, text: str) -> str:
        """
        Elimina stopwords del texto.
        
        Args:
            text: texto a procesar
        
        Returns:
            str: texto sin stopwords
        
        Ejemplo:
            Input: "the software engineer works"
            Output: "software engineer works"
        """
        words = text.split()
        return ' '.join([w for w in words if w not in self.stopwords])

    def extract_sentences(self, text: str) -> list:
        """
        Divide el texto en oraciones.
        
        Args:
            text: texto completo
        
        Returns:
            list: lista de oraciones
        
        Dependencias:
            - re.split: división con regex
        """
        return re.split(r'[.!?]+', text)

    def get_tokens(self, text: str) -> list:
        """
        Tokenización básica (separar por espacios).
        
        Args:
            text: texto a tokenizar
        
        Returns:
            list: lista de palabras
        """
        return text.split()

    def clean_resume(self, text: str) -> str:
        """
        Limpia texto de currículum línea por línea.
        
        Proceso:
        1. Dividir por líneas
        2. Eliminar líneas vacías o muy cortas (< 3 caracteres)
        3. Unir en un solo texto
        
        Args:
            text: texto del CV con saltos de línea
        
        Returns:
            str: texto limpio continuo
        
        Útil para:
            - Eliminar headers kosong
            - Normalizar formato
        """
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 2:  # Ignorar líneas muy cortas
                cleaned_lines.append(line)
        return ' '.join(cleaned_lines)