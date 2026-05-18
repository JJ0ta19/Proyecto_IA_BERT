import re
import spacy
from typing import Dict, List, Tuple, Optional

class NLPProcessor:
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model])
            self.nlp = spacy.load(model)
        
        self.nlp.max_length = 2000000
    
    def process(self, text: str) -> Dict:
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
        entities = {'PERSON': [], 'ORG': [], 'GPE': [], 'DATE': [], 'SKILL': [], 'TECH': [], 'EDUCATION': [], 'LOCATION': []}
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        tech_patterns = [
            r'\b(Python|Java|JavaScript|C\+\+|C#|Ruby|Go|Rust|Kotlin|Swift)\b',
            r'\b(Django|Flask|Spring|Rails|Express|Nest)\b',
            r'\b(React|Vue|Angular|Node|Express)\b',
            r'\b(SQL|MongoDB|Redis|PostgreSQL|MySQL|Oracle)\b',
            r'\b(AWS|GCP|Azure|Docker|Kubernetes)\b',
            r'\b(Git|GitHub|Jira|Docker|Terraform)\b',
            r'\b(TensorFlow|PyTorch|Keras|Scikit)\b',
            r'\b(HTML|CSS|Bootstrap|Tailwind)\b'
        ]
        
        text_lower = doc.text.lower()
        found_tech = set()
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            found_tech.update([m.title() for m in matches])
        
        entities['TECH'] = list(found_tech)
        
        edu_keywords = ['university', 'college', 'degree', 'bachelor', 'master', 'phd', 'certificate', 'diploma', 'education']
        education_entities = []
        for sent in doc.sents:
            if any(kw in sent.text.lower() for kw in edu_keywords):
                education_entities.append(sent.text.strip())
        entities['EDUCATION'] = education_entities[:5]
        
        return {k: list(set(v))[:20] for k, v in entities.items()}
    
    def _get_lemmas(self, doc) -> List[str]:
        return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    def _get_tokens(self, doc) -> List[Dict]:
        return [{'text': token.text, 'lemma': token.lemma_, 'pos': token.pos_, 'tag': token.tag_} for token in doc]
    
    def _get_sentences(self, doc) -> List[str]:
        return [sent.text.strip() for sent in doc.sents]
    
    def _get_pos_tags(self, doc) -> Dict:
        pos_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0, 'PROPN': 0, 'NUM': 0}
        for token in doc:
            if token.pos_ in pos_counts:
                pos_counts[token.pos_] += 1
        return pos_counts
    
    def _get_noun_chunks(self, doc) -> List[str]:
        return [chunk.text for chunk in doc.noun_chunks][:30]
    
    def _get_dependencies(self, doc) -> List[Dict]:
        return [{'text': token.text, 'dep': token.dep_, 'head': token.head.text} for token in doc[:50]]
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        section_patterns = {
            'experience': r'(?:experience|work history|employment|professional|career)',
            'education': r'(?:education|academic|degree|university|college|studies)',
            'skills': r'(?:skills|technologies|competencies|expertise|proficiencies)',
            'projects': r'(?:projects|portfolio|achievements)',
            'contact': r'(?:contact|email|phone|address|location)'
        }
        
        sections = {}
        text_lower = text.lower()
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                start = match.start()
                next_match = re.search(pattern, text_lower[start+10:])
                if next_match:
                    end = start + 10 + next_match.start()
                else:
                    end = len(text)
                sections[section_name] = text[start:end].strip()
            else:
                sections[section_name] = ""
        
        return sections
    
    def get_skills_from_ner(self, text: str) -> List[str]:
        result = self.process(text)
        skills = set()
        skills.update(result['entities'].get('TECH', []))
        
        tech_keywords = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'kotlin', 'swift',
            'html', 'css', 'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'docker', 'kubernetes',
            'aws', 'azure', 'gcp', 'git', 'linux', 'react', 'angular', 'vue', 'node', 'django', 'flask',
            'tensorflow', 'pytorch', 'keras', 'scikit', 'pandas', 'numpy', 'machine learning',
            'deep learning', 'nlp', 'data science', 'analytics', 'statistics'
        ]
        
        text_lower = text.lower()
        for kw in tech_keywords:
            if kw in text_lower:
                skills.add(kw.title())
        
        return list(skills)


nlp_processor = None

def get_nlp_processor() -> NLPProcessor:
    global nlp_processor
    if nlp_processor is None:
        nlp_processor = NLPProcessor()
    return nlp_processor