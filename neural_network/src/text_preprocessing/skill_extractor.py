import re
from typing import List, Dict, Set

class SkillExtractor:
    def __init__(self, skills_list: List[str] = None):
        self.skills_list = skills_list or []
        self._build_skill_patterns()

    def _build_skill_patterns(self):
        self.tech_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'typescript', 'php', 'swift', 'kotlin'],
            'frontend': ['html', 'css', 'react', 'angular', 'vue', 'bootstrap', 'tailwind', 'sass', 'jquery'],
            'backend': ['nodejs', 'django', 'flask', 'spring', 'rails', 'express', 'laravel', 'asp.net'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'elasticsearch', 'cassandra'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'cloudformation'],
            'data_science': ['tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'keras', 'spark', 'tableau'],
            'devops': ['jenkins', 'gitlab', 'github', 'ansible', 'puppet', 'circleci', 'travis', 'nginx'],
            'ai_ml': ['machine learning', 'deep learning', 'nlp', 'computer vision', 'neural networks', 'ai'],
        }
        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
            'creativity', 'adaptability', 'time management', 'organization',
            'critical thinking', 'decision making', 'conflict resolution',
            'project management', 'agile', 'scrum', 'mentoring', 'presentation'
        ]
        self.languages = [
            'english', 'spanish', 'french', 'german', 'chinese', 'japanese',
            'korean', 'portuguese', 'italian', 'russian', 'arabic', 'hindi'
        ]

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        text_lower = text.lower()
        skills = {
            'technical': [],
            'soft': [],
            'languages': [],
            'frameworks': [],
            'tools': [],
            'databases': [],
            'cloud': []
        }
        found_skills = set()

        for category, keywords in self.tech_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if category in ['frontend', 'backend'] and keyword not in found_skills:
                        skills['frameworks'].append(keyword)
                        found_skills.add(keyword)
                    elif category == 'database' and keyword not in found_skills:
                        skills['databases'].append(keyword)
                        found_skills.add(keyword)
                    elif category == 'cloud' and keyword not in found_skills:
                        skills['cloud'].append(keyword)
                        found_skills.add(keyword)
                    elif category == 'data_science' or category == 'ai_ml':
                        if keyword not in found_skills:
                            skills['technical'].append(keyword)
                            found_skills.add(keyword)
                    else:
                        if keyword not in found_skills:
                            skills['technical'].append(keyword)
                            found_skills.add(keyword)

        if self.skills_list:
            for skill in self.skills_list:
                skill_lower = skill.lower()
                if skill_lower in text_lower and skill_lower not in found_skills:
                    skills['technical'].append(skill)
                    found_skills.add(skill_lower)

        for soft_skill in self.soft_skills:
            if soft_skill in text_lower:
                skills['soft'].append(soft_skill)

        for language in self.languages:
            if language in text_lower:
                skills['languages'].append(language)

        return {k: list(set(v)) for k, v in skills.items()}

    def extract_from_resume(self, text: str) -> List[str]:
        all_skills = []
        extracted = self.extract_skills(text)
        for category_skills in extracted.values():
            all_skills.extend(category_skills)
        return list(set(all_skills))