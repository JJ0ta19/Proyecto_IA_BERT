"""
=================================================================
PROCESADOR NLP AVANZADO (NER, LEMMATIZACIÓN, EXTRACCIÓN DE SECCIONES)
Universidad - Pipeline de Procesamiento de Lenguaje Natural

Este módulo implementa un pipeline completo de NLP usando spaCy
para análisis de currículums.

Funcionalidades:
1. NER (Named Entity Recognition): Detectar personas, organizaciones, fechas
2. Detección de tecnologías: Python, Java, AWS, Docker, etc.
3. Extracción de secciones: Experience, Education, Skills, Projects
4. Lemmatización: Reducir palabras a su forma raíz
5. POS tagging: Etiquetado de categorías gramaticales
6. Dependency parsing: Análisis sintáctico

Dependencias:
- spacy: Framework de NLP industrial
- re: Expresiones regulares para detección de tecnologías

=================================================================
"""

import re                      # Expresiones regulares para detección de tech
import spacy                  # Framework de NLP (spaCy)
from typing import Dict, List, Tuple, Optional  # Type hints


class NLPProcessor:
    """
    Procesador completo de NLP para currículums.
    
    Utiliza spaCy con el modelo en_core_web_sm para:
    - Reconocimiento de entidades nombradas (NER)
    - Lemmatización
    - POS tagging
    - Análisis de dependencias
    
    Adicionalmente detecta:
    - Tecnologías (Python, Java, AWS, etc.) via regex
    - Secciones del CV (experience, education, skills)
    - Skills técnicos
    
    Args:
        model: nombre del modelo de spaCy (default: en_core_web_sm)
    
    Attributes:
        nlp: instancia del pipeline de spaCy
    """
    
    def __init__(self, model: str = "en_core_web_sm"):
        """
        Inicializa el procesador de NLP.
        
        Proceso:
        1. Intenta cargar el modelo de spaCy
        2. Si no existe, lo descarga automáticamente
        3. Configura el límite de longitud máxima
        
        Dependencias:
            - spacy.load: cargar modelo pre-entrenado
            - subprocess: para descargar modelo si no existe
        
        Modelo utilizado:
        - en_core_web_sm: modelo pequeño de inglés (12 MB)
          - NER: PERSON, ORG, GPE, DATE, LOC, etc.
          - POS tags: NOUN, VERB, ADJ, etc.
          - Lemmatización: inglés
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            # Si el modelo no está instalado, descargarlo
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model])
            self.nlp = spacy.load(model)
        
        # Aumentar límite de longitud para textos largos
        self.nlp.max_length = 2000000
    
    def process(self, text: str) -> Dict:
        """
        Procesa un texto y retorna información NLP completa.
        
        Pipeline:
        1. Truncar a 100k caracteres (límite de spaCy)
        2. Procesar con el pipeline de spaCy
        3. Extraer: entidades, lemmas, tokens, oraciones, POS, noun chunks, dependencias
        
        Args:
            text: texto del currículum
        
        Returns:
            dict: {
                'entities': {PERSON: [], ORG: [], TECH: [], ...},
                'lemmatized': [palabras lematizadas],
                'tokens': [{text, lemma, pos, tag}, ...],
                'sentences': [oraciones],
                'pos_tags': {NOUN: count, VERB: count, ...},
                'noun_chunks': [grupos nominales],
                'dependencies': [{text, dep, head}, ...]
            }
        
        Dependencias internas:
            - _extract_entities: NER
            - _get_lemmas: lematización
            - _get_tokens: tokens con metadatos
            - _get_sentences: segmentación
            - _get_pos_tags: conteo de POS
            - _get_noun_chunks: grupos nominales
            - _get_dependencies: análisis sintáctico
        """
        # Limitar longitud del texto (spaCy tiene límite de ~1M caracteres)
        doc = self.nlp(text[:100000])
        
        return {
            'entities': self._extract_entities(doc),
            'lemmatized': self._get_lemmas(doc),
            'tokens': self._get_tokens(doc),
            'sentences': self._get_sentences(doc),
            'pos_tags': self._get_pos_tags(doc),
            'noun_chunks': self._get_noun_chunks(doc),
            'dependencies': self._get_dependencies(doc)
        }
    
    def _extract_entities(self, doc) -> Dict:
        """
        Extrae entidades nombradas y detecta tecnologías.
        
        Tipos de entidades detectadas:
        - PERSON: nombres de personas
        - ORG: organizaciones (empresas)
        - GPE: países/ciudades
        - DATE: fechas
        - TECH: tecnologías (detección personalizada con regex)
        - EDUCATION: secciones educativas (detección por palabras clave)
        
        Proceso:
        1. Iterar sobre entidades de spaCy (doc.ents)
        2. Detectar tecnologías con patrones regex
        3. Detectar educación con palabras clave
        
        Args:
            doc: documento procesado por spaCy
        
        Returns:
            dict: entidades por tipo
        
        Dependencias:
            - re.findall: búsqueda de patrones
            - doc.ents: iterator de entidades spaCy
            - ent.label_: tipo de entidad
            - ent.text: texto de la entidad
        """
        # Inicializar diccionario de entidades
        entities = {'PERSON': [], 'ORG': [], 'GPE': [], 'DATE': [], 
                   'SKILL': [], 'TECH': [], 'EDUCATION': [], 'LOCATION': []}
        
        # Extraer entidades del modelo spaCy
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        # =================================================================
        # DETECCIÓN DE TECNOLOGÍAS (Patrones Regex)
        # =================================================================
        # SpaCy no tiene entidad TECH, así que usamos regex
        
        # Lista de patrones de tecnologías conocidas
        tech_patterns = [
            # Lenguajes de programación
            r'\b(Python|Java|JavaScript|C\+\+|C#|Ruby|Go|Rust|Kotlin|Swift)\b',
            # Frameworks web
            r'\b(Django|Flask|Spring|Rails|Express|Nest)\b',
            # Frontend
            r'\b(React|Vue|Angular|Node|Express)\b',
            # Bases de datos
            r'\b(SQL|MongoDB|Redis|PostgreSQL|MySQL|Oracle)\b',
            # Cloud y DevOps
            r'\b(AWS|GCP|Azure|Docker|Kubernetes)\b',
            # Herramientas
            r'\b(Git|GitHub|Jira|Docker|Terraform)\b',
            # ML/AI
            r'\b(TensorFlow|PyTorch|Keras|Scikit)\b',
            # Web básico
            r'\b(HTML|CSS|Bootstrap|Tailwind)\b'
        ]
        
        text_lower = doc.text.lower()
        found_tech = set()
        
        # Buscar cada patrón en el texto
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            found_tech.update([m.title() for m in matches])
        
        entities['TECH'] = list(found_tech)
        
        # =================================================================
        # DETECCIÓN DE EDUCACIÓN
        # =================================================================
        # Buscar oraciones que contengan palabras relacionadas con educación
        edu_keywords = ['university', 'college', 'degree', 'bachelor', 'master', 
                        'phd', 'certificate', 'diploma', 'education']
        
        education_entities = []
        for sent in doc.sents:
            # Si la oración contiene palabras clave educativas
            if any(kw in sent.text.lower() for kw in edu_keywords):
                education_entities.append(sent.text.strip())
        
        entities['EDUCATION'] = education_entities[:5]  # Máximo 5
        
        # Limitar a 20 entidades por tipo
        return {k: list(set(v))[:20] for k, v in entities.items()}
    
    def _get_lemmas(self, doc) -> List[str]:
        """
        Extrae lemas (forma raíz de las palabras).
        
        Proceso:
        1. Iterar sobre todos los tokens
        2. Excluir stopwords y puntuación
        3. Retornar lemma de cada palabra
        
        Args:
            doc: documento spaCy
        
        Returns:
            list: palabras en su forma raíz
        
        Ejemplo:
            "running" -> "run"
            "developed" -> "develop"
            "technologies" -> "technology"
        
        Dependencias:
            - token.lemma_: forma lematizada
            - token.is_stop: es stopword?
            - token.is_punct: es puntuación?
        """
        return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    def _get_tokens(self, doc) -> List[Dict]:
        """
        Extrae tokens con metadatos.
        
        Args:
            doc: documento spaCy
        
        Returns:
            list: [{text, lemma, pos, tag}, ...]
        
        Campos:
            - text: texto original
            - lemma: forma raíz
            - pos: categoría gramatical (NOUN, VERB, etc.)
            - tag: etiqueta detallada (NN, VB, etc.)
        """
        return [{'text': token.text, 'lemma': token.lemma_, 
                 'pos': token.pos_, 'tag': token.tag_} for token in doc]
    
    def _get_sentences(self, doc) -> List[str]:
        """
        Segmenta el texto en oraciones.
        
        Args:
            doc: documento spaCy
        
        Returns:
            list: oraciones del texto
        """
        return [sent.text.strip() for sent in doc.sents]
    
    def _get_pos_tags(self, doc) -> Dict:
        """
        Cuenta las categorías gramaticales (POS tags).
        
       POS Tags principales:
        - NOUN: sustantivos
        - VERB: verbos
        - ADJ: adjetivos
        - ADV: adverbios
        - PROPN: nombres propios
        - NUM: números
        
        Args:
            doc: documento spaCy
        
        Returns:
            dict: {POS: cantidad}
        """
        pos_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0, 'PROPN': 0, 'NUM': 0}
        for token in doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1
        return pos_counts
    
    def _get_noun_chunks(self, doc) -> List[str]:
        """
        Extrae grupos nominales (noun chunks).
        
        Un noun chunk es una frase nominal: "the modern software engineer"
        
        Args:
            doc: documento spaCy
        
        Returns:
            list: grupos nominales
        """
        return [chunk.text for chunk in doc.noun_chunks][:30]
    
    def _get_dependencies(self, doc) -> List[Dict]:
        """
        Extrae análisis de dependencias sintácticas.
        
        Muestra la relación entre palabras:
        - nsubj: sujeto nominal
        - dobj: objeto directo
        - prep: preposición
        
        Args:
            doc: documento spaCy
        
        Returns:
            list: [{text, dep, head}, ...]
        """
        return [{'text': token.text, 'dep': token.dep_, 
                 'head': token.head.text} for token in doc[:50]]
    
    # =================================================================
    # EXTRACCIÓN DE SECCIONES DEL CV
    # =================================================================
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extrae secciones relevantes del currículum.
        
        Secciones detectadas:
        - experience: experiencia laboral
        - education: formación académica
        - skills: habilidades técnicas
        - projects: proyectos
        - contact: información de contacto
        
        Proceso:
        1. Buscar patrones de cada sección en el texto
        2. Extraer texto entre patrones
        3. Manejar casos donde no se encuentra ninguna sección
        
        Args:
            text: texto completo del CV
        
        Returns:
            dict: {sección: contenido}
        """
        # Patrones regex para identificar secciones
        section_patterns = {
            'experience': r'(?:experience|work history|employment|professional|career)',
            'education': r'(?:education|academic|degree|university|college|studies)',
            'skills': r'(?:skills|technologies|competencies|expertise|proficiencies)',
            'projects': r'(?:projects|portfolio|achievements)',
            'contact': r'(?:contact|email|phone|address|location)'
        }
        
        sections = {}
        text_lower = text.lower()
        
        # Buscar cada sección
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                start = match.start()
                # Buscar el inicio de la siguiente sección
                next_match = re.search(pattern, text_lower[start+10:])
                if next_match:
                    end = start + 10 + next_match.start()
                else:
                    end = len(text)
                sections[section_name] = text[start:end].strip()
            else:
                sections[section_name] = ""
        
        return sections
    
    # =================================================================
    # EXTRACCIÓN DE SKILLS
    # =================================================================
    
    def get_skills_from_ner(self, text: str) -> List[str]:
        """
        Extrae habilidades técnicas del texto.
        
        Combina:
        - Entidades TECH detectadas por regex
        - Palabras clave de tecnologías en el texto
        
        Args:
            text: texto del CV
        
        Returns:
            list: skills detectados (únicos)
        """
        # Usar el método de extracción de entidades
        result = self.process(text)
        skills = set()
        
        # Agregar tecnologías de NER
        skills.update(result['entities'].get('TECH', []))
        
        # Lista de keywords de technologies known
        tech_keywords = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 
            'kotlin', 'swift', 'html', 'css', 'sql', 'mysql', 'postgresql', 
            'mongodb', 'redis', 'docker', 'kubernetes', 'aws', 'azure', 
            'gcp', 'git', 'linux', 'react', 'angular', 'vue', 'node', 
            'django', 'flask', 'tensorflow', 'pytorch', 'keras', 'scikit', 
            'pandas', 'numpy', 'machine learning', 'deep learning', 
            'nlp', 'data science', 'analytics', 'statistics'
        ]
        
        text_lower = text.lower()
        for kw in tech_keywords:
            if kw in text_lower:
                skills.add(kw.title())
        
        return list(skills)


# ====================================================================
# INSTANCIA GLOBAL (Singleton)
# ====================================================================

# Instancia global del procesador NLP
# Se inicializa una sola vez para evitar recargar el modelo
nlp_processor = None

def get_nlp_processor() -> NLPProcessor:
    """
    Retorna la instancia global del procesador NLP.
    
    Implementa patrón Singleton para evitar:
    - Carga repetida del modelo de spaCy
    - Alto consumo de memoria
    
    Returns:
        NLPProcessor: instancia única del procesador
    """
    global nlp_processor
    if nlp_processor is None:
        nlp_processor = NLPProcessor()
    return nlp_processor