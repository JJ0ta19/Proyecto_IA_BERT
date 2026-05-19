"""
=================================================================
MODELO BERT PARA CLASIFICACIÓN DE CURRÍCULUMS
Universidad - Clasificador de Hojas de Vida

Este módulo define la arquitectura del modelo BERT personalizado
para clasificación de currículums en 42 categorías profesionales.

Arquitectura:
- BERT base (bert-base-uncased): 12 capas transformer, 768 hidden, 12 heads
- Dropout: 30% para evitar overfitting
- Capa lineal: 768 -> 42 (número de categorías)

=================================================================
"""

# ====================================================================
# IMPORTS
# ====================================================================

import torch                    # Framework de deep learning (PyTorch)
import torch.nn as nn           # Módulos de redes neuronales
from torch.utils.data import Dataset, DataLoader  # Datos y batches
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup  # BERT
from torch.optim import AdamW   # Optimizador Adam con weight decay
from typing import Optional     # Type hints opcionales
import numpy as np              # Operaciones numéricas


# ====================================================================
# CLASE: ResumeDataset
# ====================================================================

class ResumeDataset(Dataset):
    """
    Dataset personalizado para currículums.
    
    Hereda de PyTorch Dataset para integrar con DataLoader.
    Convierte textos y etiquetas en tensores para BERT.
    
    Args:
        texts: lista de textos de currículums
        labels: lista de etiquetas (números)
        tokenizer: tokenizador de BERT
        max_len: longitud máxima de tokens (default: 512)
    
    Dependencias:
        - torch.utils.data.Dataset: clase base de dataset
        - transformers.BertTokenizer: tokenización de texto
    """
    
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts           # Lista de textos
        self.labels = labels        # Lista de etiquetas (números)
        self.tokenizer = tokenizer  # Tokenizador de BERT
        self.max_len = max_len      # Longitud máxima

    def __len__(self):
        """Retorna el número de samples en el dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retorna un sample tokenizado.
        
        Proceso:
        1. Obtener texto y etiqueta
        2. Tokenizar con BERT tokenizer
        3. Retornar dict con input_ids, attention_mask y labels
        
        Returns:
            dict: {
                'input_ids': tensor de tokens,
                'attention_mask': tensor de máscara,
                'labels': tensor de etiqueta
            }
        """
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenizar texto con BERT
        # add_special_tokens: agregar [CLS] y [SEP]
        # padding: rellenor a longitud máxima
        # truncation: truncar si es muy largo
        # return_attention_mask: máscara de atención
        # return_tensors: retornar tensores de PyTorch
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


# ====================================================================
# CLASE: BertClassifier
# ====================================================================

class BertClassifier(nn.Module):
    """
    Arquitectura de red neuronal para clasificación.
    
    Arquitectura:
    1. BERT base (12 layers, 768 hidden, 12 attention heads)
    2. Dropout (30%)
    3. Capa lineal (768 -> num_classes)
    
    Args:
        num_classes: número de categorías de salida
        model_name: nombre del modelo pre-entrenado (default: bert-base-uncased)
        dropout: tasa de dropout (default: 0.3)
    
    Dependencias:
        - torch.nn.Module: clase base de PyTorch
        - transformers.BertModel: modelo BERT pre-entrenado
        - torch.nn.Dropout: capa de regularización
        - torch.nn.Linear: capa fully-connected
    """
    
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased", dropout: float = 0.3):
        super(BertClassifier, self).__init__()
        
        # Cargar BERT pre-entrenado ( pesos de Google )
        # bert-base-uncased: 12 capas, 768 hidden, 12 heads, 110M parámetros
        self.bert = BertModel.from_pretrained(model_name)
        
        # Dropout para regularización (evita overfitting)
        self.dropout = nn.Dropout(dropout)
        
        # Capa de clasificación: 768 dimensiones -> num_classes
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass de la red.
        
        Proceso:
        1. Pasar input_ids y attention_mask por BERT
        2. Obtener pooled output (cls token)
        3. Aplicar dropout
        4. Pasar por capa lineal
        
        Args:
            input_ids: tensores de tokens (batch_size, seq_len)
            attention_mask: máscara de atención (batch_size, seq_len)
        
        Returns:
            torch.Tensor: logits de clasificación (batch_size, num_classes)
        """
        # BERT retorna dos outputs:
        # - last_hidden_state: (batch, seq_len, 768)
        # - pooler_output: (batch, 768) - CLS token procesado
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Tomar el CLS token
        
        # Aplicar dropout para regularización
        pooled_output = self.dropout(pooled_output)
        
        # Capa lineal de clasificación
        return self.classifier(pooled_output)


# ====================================================================
# CLASE: BertClassifierModel
# ====================================================================

class BertClassifierModel:
    """
    Clase principal que encapsula el modelo BERT y sus operaciones.
    
    Métodos:
    - __init__: inicializar modelo y tokenizador
    - train: entrenar el modelo con datos de entrenamiento
    - predict: predecir clase única
    - predict_proba: obtener probabilidades de todas las clases
    - save_model: guardar modelo a archivo
    - load_model: cargar modelo desde archivo
    
    Args:
        num_classes: número de categorías
        model_name: nombre del modelo BERT
        device: 'cuda' o 'cpu'
    """
    
    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased", device: str = "cuda"):
        # Seleccionar dispositivo (GPU si está disponible, sino CPU)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Crear modelo y mover al dispositivo
        self.model = BertClassifier(num_classes=num_classes, model_name=model_name).to(self.device)
        
        # Cargar tokenizador de BERT
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Guardar número de clases
        self.num_classes = num_classes

    def train(self, train_texts, train_labels, val_texts, val_labels,
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5,
              max_len: int = 512, save_path: Optional[str] = None):
        """
        Entrena el modelo BERT.
        
        Pipeline de entrenamiento:
        1. Crear datasets de entrenamiento y validación
        2. Crear DataLoaders
        3. Configurar optimizador (AdamW) y scheduler
        4. Loop de epochs: entrenamiento + evaluación
        5. Guardar modelo si se especifica
        
        Args:
            train_texts: textos de entrenamiento
            train_labels: etiquetas de entrenamiento
            val_texts: textos de validación
            val_labels: etiquetas de validación
            epochs: número de épocas
            batch_size: tamaño de batch
            learning_rate: tasa de aprendizaje
            max_len: longitud máxima de tokens
            save_path: ruta para guardar modelo (opcional)
        
        Dependencias:
            - ResumeDataset: dataset personalizado
            - DataLoader: batching de datos
            - AdamW: optimizador
            - get_linear_schedule_with_warmup: scheduler de learning rate
            - nn.CrossEntropyLoss: función de pérdida
        """
        # Crear datasets
        train_dataset = ResumeDataset(train_texts, train_labels, self.tokenizer, max_len)
        val_dataset = ResumeDataset(val_texts, val_labels, self.tokenizer, max_len)

        # Crear DataLoaders (iteradores de batches)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizador AdamW (Adam con weight decay para regularización)
        # learning_rate típico para BERT: 2e-5 (0.00002)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Scheduler: learning rate lineal con warmup
        # Warmup: aumentar gradualmente al inicio
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Loop de épocas
        for epoch in range(epochs):
            # Entrenar una época
            self._train_epoch(epoch + 1, train_loader, optimizer, scheduler)
            
            # Evaluar en validación
            val_acc = self._evaluate(val_loader)
            print(f"Epoch {epoch + 1} - Validation Accuracy: {val_acc:.4f}")

        # Guardar modelo al finalizar
        if save_path:
            self.save_model(save_path)
            print(f"Model saved to {save_path}")

    def _train_epoch(self, epoch, train_loader, optimizer, scheduler):
        """
        Entrena una sola época.
        
        Proceso:
        1. Poner modelo en modo entrenamiento
        2. Por cada batch:
           - Forward pass
           - Calcular pérdida
           - Backward pass
           - Clip gradientes
           - Optimizador step
           - Scheduler step
        3. Imprimir pérdida promedio
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Mover datos al dispositivo (GPU/CPU)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Limpiar gradientes previos
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            
            # Calcular pérdida (Cross Entropy para multi-class)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # Backward pass (calcular gradientes)
            loss.backward()
            
            # Clip de gradientes (evitar explosiones de gradiente)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Actualizar pesos
            optimizer.step()
            
            # Actualizar learning rate (scheduler)
            scheduler.step()

            total_loss += loss.item()
            
            # Imprimir cada 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()::.4f}")

        print(f"Epoch {epoch} - Avg Loss: {total_loss / len(train_loader):.4f}")

    def _evaluate(self, val_loader):
        """
        Evalúa el modelo en el conjunto de validación.
        
        Proceso:
        1. Poner modelo en modo evaluación
        2. Por cada batch: hacer predicciones y contar aciertos
        3. Calcular accuracy
        
        Returns:
            float: accuracy (correctos / total)
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():  # No calcular gradientes (más rápido)
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                
                # Predicción: índice con mayor probabilidad
                preds = torch.argmax(outputs, dim=1)
                
                # Contar aciertos
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0

    def predict(self, texts, batch_size: int = 16, max_len: int = 512):
        """
        Predice la categoría para una lista de textos.
        
        Args:
            texts: lista de textos a clasificar
            batch_size: tamaño de batch
            max_len: longitud máxima de tokens
        
        Returns:
            list: índices de las clases predichas
        
        Dependencias:
            - ResumeDataset: para tokenizar textos
            - DataLoader: para procesar en batches
            - torch.argmax: obtener índice máximo
        """
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
        """
        Predice probabilidades para cada categoría.
        
        Args:
            texts: lista de textos
            batch_size: tamaño de batch
            max_len: longitud máxima
        
        Returns:
            np.ndarray: probabilidades (num_texts, num_classes)
        
        Dependencias:
            - torch.softmax: convertir logits a probabilidades
            - np.vstack: apilar arrays
        """
        self.model.eval()
        dataset = ResumeDataset(texts, [0] * len(texts), self.tokenizer, max_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                
                # Softmax para obtener probabilidades (suma 1)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def save_model(self, path: str):
        """
        Guarda el modelo a un archivo .pt
        
        Args:
            path: ruta del archivo
        
        Dependencias:
            - torch.save: guardar diccionario con state_dict
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),  # Pesos de la red
            'num_classes': self.num_classes               # Número de clases
        }, path)

    def load_model(self, path: str):
        """
        Carga el modelo desde un archivo .pt
        
        Args:
            path: ruta del archivo
        
        Dependencias:
            - torch.load: cargar diccionario con state_dict
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.num_classes = checkpoint['num_classes']