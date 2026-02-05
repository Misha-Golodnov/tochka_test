import sys
import os

sys.path.append('..')

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

from src.data_preprocessing import TextQualityDataset, load_and_prepare_data

# Конфигурация
MODEL_NAME = "cointegrated/rubert-tiny2"
BATCH_SIZE = 8
EPOCHS = 1  # Только 1 эпоха для теста


def train_simple_model(task_name, train_texts, train_labels, val_texts, val_labels):
    """Упрощенное обучение модели"""

    print(f"\nОбучение модели для {task_name}...")

    # Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Датасеты
    train_dataset = TextQualityDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextQualityDataset(val_texts, val_labels, tokenizer)

    # Даталоадеры
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Модель
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    model.to(device)

    # Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Обучение (упрощенное)
    model.train()
    for epoch in range(EPOCHS):
        print(f"Эпоха {epoch + 1}/{EPOCHS}")

        for batch_idx, batch in enumerate(train_loader):
            # Перемещаем батч на устройство
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                print(f"  Батч {batch_idx}, Loss: {loss.item():.4f}")

    # Валидация
    model.eval()
    val_predictions = []
    val_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=1)
            val_predictions.extend(predictions.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    accuracy = accuracy_score(val_true, val_predictions)
    print(f"Accuracy на валидации: {accuracy:.4f}")

    return model, tokenizer


def main():
    print("Быстрое обучение на небольшом датасете...")

    # Загружаем данные
    tasks_data = load_and_prepare_data("../data/train_sample.parquet")

    models = {}
    tokenizers = {}

    # Обучаем модели для каждой задачи
    for task in ['integrity', 'factuality', 'truthfulness']:
        task_data = tasks_data[task]

        model, tokenizer = train_simple_model(
            task_name=task,
            train_texts=task_data['train_texts'],
            train_labels=task_data['train_labels'],
            val_texts=task_data['val_texts'],
            val_labels=task_data['val_labels']
        )

        models[task] = model
        tokenizers[task] = tokenizer

    print("\n✅ Обучение завершено!")
    return models, tokenizers


if __name__ == "__main__":
    models, tokenizers = main()