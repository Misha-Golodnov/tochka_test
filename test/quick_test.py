import sys
import os
sys.path.append('..')

import pandas as pd
import torch
from transformers import AutoTokenizer

# 1. Загрузим небольшую часть данных для теста
print("Загрузка небольшого подмножества...")
train_full = pd.read_parquet("../data/train.parquet")
test_full = pd.read_parquet("../data/test.parquet")

# Возьмем только 100 примеров для быстрого теста
train_sample = train_full.head(100).copy()
test_sample = test_full.head(20).copy()

# Сохраним временные файлы
train_sample.to_parquet("data/train_sample.parquet")
test_sample.to_parquet("data/test_sample.parquet")

print(f"Созданы тестовые файлы:")
print(f"- train_sample.parquet: {len(train_sample)} примеров")
print(f"- test_sample.parquet: {len(test_sample)} примеров")

# 2. Тест датасета
print("\nТест класса TextQualityDataset...")
from src.data_preprocessing import TextQualityDataset

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

# Создаем датасет
texts = ["Пример текста 1", "Пример текста 2"]
labels = [0, 1]
dataset = TextQualityDataset(texts, labels, tokenizer)

print(f"✓ Dataset создан, размер: {len(dataset)}")
print(f"✓ Пример элемента: {dataset[0].keys()}")

# 3. Тест предобработки
print("\nТест функции load_and_prepare_data...")
from src.data_preprocessing import load_and_prepare_data

tasks_data = load_and_prepare_data("../data/train_sample.parquet")
print(f"✓ Данные подготовлены для {len(tasks_data)} задач")
for task, data in tasks_data.items():
    print(f"  {task}: {len(data['train_texts'])} train, {len(data['val_texts'])} val")

print("\n✅ Все тесты пройдены!")