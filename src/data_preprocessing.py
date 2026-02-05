import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class TextQualityDataset(Dataset):
    """Датасет для классификации качества текстов"""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def load_and_prepare_data(train_path, test_path=None):
    """Загрузка и подготовка данных"""

    # Загрузка train данных
    train_df = pd.read_parquet(train_path)

    # Фильтруем только 0 и 1 (игнорируем 0.5)
    mask = (
            train_df['integrity'].isin([0, 1]) &
            train_df['factuality'].isin([0, 1]) &
            train_df['truthfulness'].isin([0, 1])
    )
    train_df = train_df[mask].copy()

    print(f"Train данных после фильтрации 0.5: {len(train_df)}")

    # Подготовка данных для каждой задачи
    tasks_data = {}

    for task in ['integrity', 'factuality', 'truthfulness']:
        # Используем текст + reasoning для лучшего качества
        texts = []
        for idx, row in train_df.iterrows():
            text_with_reasoning = f"Текст: {row['text']}\n\n"
            if pd.notna(row.get(f'{task}_reasoning')):
                text_with_reasoning += f"Объяснение: {row[f'{task}_reasoning']}"
            texts.append(text_with_reasoning)

        labels = train_df[task].astype(int).values

        # Разделение на train/val
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=0.1,
            random_state=42,
            stratify=labels
        )

        tasks_data[task] = {
            'train_texts': train_texts,
            'train_labels': train_labels,
            'val_texts': val_texts,
            'val_labels': val_labels
        }

        print(f"{task}: train={len(train_texts)}, val={len(val_texts)}")

    # Загрузка test данных если нужно
    test_texts = None
    if test_path:
        test_df = pd.read_parquet(test_path)
        test_texts = test_df['text'].tolist()
        test_uuids = test_df.index.tolist()
        return tasks_data, test_texts, test_uuids

    return tasks_data

def load_test_data(test_path):
    """Загрузка только тестовых данных"""
    test_df = pd.read_parquet(test_path)
    test_texts = test_df['text'].tolist()
    test_uuids = test_df.index.tolist()
    return test_texts, test_uuids