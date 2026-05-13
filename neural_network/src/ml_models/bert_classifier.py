import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from typing import Optional
import numpy as np

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
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

class BertClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased", dropout: float = 0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class BertClassifierModel:
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = BertClassifier(num_classes=num_classes, model_name=model_name).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.num_classes = num_classes

    def train(self, train_texts, train_labels, val_texts, val_labels,
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5,
              max_len: int = 512, save_path: Optional[str] = None):

        train_dataset = ResumeDataset(train_texts, train_labels, self.tokenizer, max_len)
        val_dataset = ResumeDataset(val_texts, val_labels, self.tokenizer, max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(epochs):
            self._train_epoch(epoch + 1, train_loader, optimizer, scheduler)
            val_acc = self._evaluate(val_loader)
            print(f"Epoch {epoch + 1} - Validation Accuracy: {val_acc:.4f}")

        if save_path:
            self.save_model(save_path)
            print(f"Model saved to {save_path}")

    def _train_epoch(self, epoch, train_loader, optimizer, scheduler):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch} - Avg Loss: {total_loss / len(train_loader):.4f}")

    def _evaluate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0

    def predict(self, texts, batch_size: int = 16, max_len: int = 512):
        self.model.eval()
        dataset = ResumeDataset(texts, [0] * len(texts), self.tokenizer, max_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())

        return predictions

    def predict_proba(self, texts, batch_size: int = 16, max_len: int = 512):
        self.model.eval()
        dataset = ResumeDataset(texts, [0] * len(texts), self.tokenizer, max_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.num_classes = checkpoint['num_classes']