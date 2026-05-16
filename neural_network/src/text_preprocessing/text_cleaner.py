import re
import string
import unicodedata
from typing import Optional

class TextCleaner:
    def __init__(self):
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
        if not isinstance(text, str) or not text.strip():
            return ""
        text = self._normalize_unicode(text)
        text = self._expand_contractions(text)
        text = self._remove_special_chars(text)
        text = self._normalize_whitespace(text)
        text = text.lower()
        return text

    def _normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    def _expand_contractions(self, text: str) -> str:
        for key, value in self.contractions.items():
            text = re.sub(r'\b' + key + r'\b', value, text, flags=re.IGNORECASE)
        return text

    def _remove_special_chars(self, text: str) -> str:
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def remove_stopwords(self, text: str) -> str:
        words = text.split()
        return ' '.join([w for w in words if w not in self.stopwords])

    def extract_sentences(self, text: str) -> list:
        return re.split(r'[.!?]+', text)

    def get_tokens(self, text: str) -> list:
        return text.split()

    def clean_resume(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 2:
                cleaned_lines.append(line)
        return ' '.join(cleaned_lines)