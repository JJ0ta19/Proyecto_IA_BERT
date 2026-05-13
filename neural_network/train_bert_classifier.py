import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "1_resume_classification", "training_data.csv")
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['Resume Text', 'Job Role'])
    print(f"Total samples: {len(df)}")
    print(f"Job Roles: {df['Job Role'].nunique()}")
    print(f"Categories: {df['Category'].nunique()}")
    print(df['Category'].value_counts())
    return df

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def main():
    print("Loading data...")
    df = load_data()

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Job Role'])
    num_classes = len(label_encoder.classes_)
    print(f"Number of Job Roles: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Resume Text'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].values
    )

    print("\nLoading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_classes
    )
    model = model.to(device)

    train_dataset = ResumeDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = ResumeDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training loss: {train_loss:.4f}")

        val_accuracy = evaluate(model, val_loader, device)
        print(f"Validation accuracy: {val_accuracy:.4f}")

    print("\nSaving model...")
    torch.save(model.state_dict(), 'models/bert_classifier.pt')
    torch.save(label_encoder, 'models/label_encoder.pkl')
    
    # Guardar mapeo Job Role -> Category
    job_role_category_map = df[['Job Role', 'Category']].drop_duplicates()
    job_role_category_map.to_csv('models/job_role_category_map.csv', index=False)
    print("Model saved!")

if __name__ == "__main__":
    main()