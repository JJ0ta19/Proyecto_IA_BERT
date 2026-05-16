import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from typing import Optional, List, Dict
import numpy as np
import time
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

class BertClassifier(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased", dropout: float = 0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.bert.pooler = nn.Identity()
        
        for name, param in self.bert.named_parameters():
            if "layer.11" in name or "layer.10" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled_output)
        return self.classifier(x)

    def get_bert_trainable_params(self):
        return [p for n, p in self.bert.named_parameters() if p.requires_grad]

    def get_classifier_params(self):
        return list(self.classifier.parameters())

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BertClassifierModel:
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased", device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA no disponible. Se requiere GPU NVIDIA.")
        
        self.device = torch.device(device)
        print(f"[BERT] Dispositivo: {self.device}")
        if torch.cuda.is_available():
            print(f"[BERT] GPU: {torch.cuda.get_device_name(0)}")
        
        self.model = BertClassifier(num_classes=num_classes, model_name=model_name).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.num_classes = num_classes

    def train(self, train_texts, train_labels, val_texts, val_labels,
              epochs: int = 10, batch_size: int = 64, learning_rate: float = 2e-4,
              max_len: int = 256, save_path: Optional[str] = None, patience: int = 2):

        if self.device.type != 'cuda':
            raise RuntimeError("Se requiere GPU para entrenar.")
        
        print(f"\n{'='*60}")
        print(f"[TRAIN] INICIANDO ENTRENAMIENTO")
        print(f"{'='*60}")
        print(f"[TRAIN] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[TRAIN] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"[TRAIN] Config: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}")
        print(f"{'='*60}\n")
        
        train_dataset = ResumeDataset(train_texts, train_labels, self.tokenizer, max_len)
        val_dataset = ResumeDataset(val_texts, val_labels, self.tokenizer, max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"[TRAIN] Parametros entrenables: {sum(p.numel() for p in trainable_params):,}")
        print(f"[TRAIN] Batches por epoch: {len(train_loader)}")
        print(f"[TRAIN] Total steps: {len(train_loader) * epochs}")
        print()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"[TRAIN] Parametros entrenables: {sum(p.numel() for p in trainable_params):,}")
        
        bert_params = self.model.get_bert_trainable_params()
        classifier_params = self.model.get_classifier_params()
        
        optimizer = AdamW([
            {'params': bert_params, 'lr': 1e-5},
            {'params': classifier_params, 'lr': 3e-4}
        ], weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
        
        scaler = GradScaler('cuda')
        best_val_acc = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss = self._train_epoch(epoch + 1, train_loader, optimizer, scheduler, scaler)
            val_acc = self._evaluate(val_loader)
            epoch_time = time.time() - epoch_start
            acc_percent = val_acc * 100
            
            print(f"\n{'='*50}")
            print(f"[RESULTADO EPOCH {epoch + 1}]")
            print(f"{'='*50}")
            print(f"  Loss: {train_loss:.4f}")
            print(f"  Accuracy: {acc_percent:.2f}%")
            print(f"  Tiempo: {epoch_time:.1f}s")
            print(f"{'='*50}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                if save_path:
                    self.save_model(save_path)
                    print(f"[SAVE] Mejor modelo (Acc: {acc_percent:.2f}%)")
            else:
                patience_counter += 1
                print(f"[WARN] No hubo mejora en epoch {epoch + 1}")
        
        print(f"[TRAIN] Entrenamiento completado. Mejor accuracy: {best_val_acc * 100:.2f}%")

    def _train_epoch(self, epoch, train_loader, optimizer, scheduler, scaler):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        print(f"\n[EPOCH {epoch}] Entrenando...")
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=True)
        
        for batch_idx, (batch) in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = self.model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            current_loss = loss.item()
            
            elapsed_ms = (time.time() - start_time) * 1000
            batch_per_sec = (batch_idx + 1) / (time.time() - start_time) if batch_idx > 0 else 0
            
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'ms/batch': f'{elapsed_ms / (batch_idx + 1):.0f}',
                'b/s': f'{batch_per_sec:.1f}'
            })

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - start_time
        
        print(f"[EPOCH {epoch}] Completado | Loss: {avg_loss:.4f} | Tiempo: {elapsed:.1f}s")
        
        return avg_loss

    def _evaluate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                with autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0

    def predict(self, texts, batch_size: int = 16, max_len: int = 256):
        self.model.eval()
        dataset = ResumeDataset(texts, [0] * len(texts), self.tokenizer, max_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                with autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())

        return predictions

    def predict_proba(self, texts, batch_size: int = 16, max_len: int = 256):
        self.model.eval()
        dataset = ResumeDataset(texts, [0] * len(texts), self.tokenizer, max_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                with autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def save_model(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.num_classes = checkpoint['num_classes']