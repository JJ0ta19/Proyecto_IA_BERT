import pandas as pd
import numpy as np
from typing import Tuple, Optional
from ..preprocessing.text_cleaner import TextCleaner

class DatasetPreprocessor:
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.label_encoder = None

    def prepare_classification_data(self, df: pd.DataFrame, text_col: str = "Resume_str", label_col: str = "Category") -> Tuple[pd.DataFrame, pd.Series]:
        print("Preparing classification data...")
        df = df.copy()

        df = df.dropna(subset=[text_col])
        df = df[df[text_col].astype(str).str.strip() != ""]
        df = df[df[text_col].str.len() > 10]

        print(f"After dropping nulls/empty: {len(df)} samples")

        df['cleaned_text'] = df[text_col].apply(self.text_cleaner.clean)
        df = df.dropna(subset=['cleaned_text'])
        df = df[df['cleaned_text'].str.len() > 10]

        print(f"After cleaning: {len(df)} samples")

        df = df.drop_duplicates(subset=['cleaned_text'])
        print(f"After dedup: {len(df)} samples")

        self._encode_labels(df, label_col)
        return df['cleaned_text'], df['label']

    def _encode_labels(self, df: pd.DataFrame, label_col: str):
        unique_labels = df[label_col].unique()
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
        df['label'] = df[label_col].map(self.label_mapping)
        print(f"Encoded {len(unique_labels)} unique labels")

    def prepare_for_bert(self, texts: pd.Series, labels: pd.Series, val_size: float = 0.2, random_state: int = 42) -> Tuple:
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts.values, labels.values,
            test_size=val_size,
            random_state=random_state,
            stratify=labels.values
        )
        print(f"Train: {len(train_texts)}, Validation: {len(val_texts)}")
        return train_texts, val_texts, train_labels, val_labels

    def merge_datasets_for_training(self, df1: pd.DataFrame, df2: pd.DataFrame, text_col1: str = "Resume_str", text_col2: str = "Resume_str", label_col: str = "Category") -> pd.DataFrame:
        df1_subset = df1[[text_col1, label_col]].copy()
        df1_subset.columns = ['text', 'label']

        df2_subset = df2[[text_col2, label_col]].copy()
        df2_subset.columns = ['text', 'label']

        merged = pd.concat([df1_subset, df2_subset], ignore_index=True)
        merged = merged.dropna()
        merged = merged.drop_duplicates(subset=['text'])
        print(f"Merged dataset: {len(merged)} samples")
        return merged

    def get_label_mapping(self):
        return self.label_mapping, self.reverse_label_mapping