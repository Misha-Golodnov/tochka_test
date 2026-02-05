import sys
import os
sys.path.append('..')

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

print("Тест создания submission файла...")

# 1. Загрузим test данные
test_df = pd.read_parquet("../data/test_sample.parquet")
test_texts = test_df['text'].tolist()
test_uuids = test_df.index.tolist()

print(f"Test данных: {len(test_texts)}")

# 2. Создадим "фиктивную" модель для теста
print("\nСоздание фиктивной модели...")
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModelForSequenceClassification.from_pretrained(
    "cointegrated/rubert-tiny2",
    num_labels=2
)

# 3. Предсказание на нескольких примерах
print("\nПредсказание на 3 примерах...")
device = torch.device('cpu')
model.to(device)
model.eval()

sample_texts = test_texts[:3]
sample_texts_with_prompt = [f"Текст: {text[:200]}..." for text in sample_texts]

# Токенизация
encodings = tokenizer(
    sample_texts_with_prompt,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='pt'
).to(device)

# Предсказание
with torch.no_grad():
    outputs = model(
        input_ids=encodings['input_ids'],
        attention_mask=encodings['attention_mask']
    )
    predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

print(f"Предсказания: {predictions}")

# 4. Создание submission файла
print("\nСоздание submission.csv...")
import numpy as np

# Создаем случайные предсказания для всех тестовых данных
np.random.seed(42)
predictions_all = {
    'integrity': np.random.randint(0, 2, len(test_texts)),
    'factuality': np.random.randint(0, 2, len(test_texts)),
    'truthfulness': np.random.randint(0, 2, len(test_texts))
}

# Создаем DataFrame
submission = pd.DataFrame({
    'uuid': test_uuids,
    'integrity': predictions_all['integrity'],
    'truthfulness': predictions_all['truthfulness'],
    'factuality': predictions_all['factuality']
})

# Сохраняем
submission.to_csv('submission_sample.csv', index=False)
print(f"Файл сохранен: submission_sample.csv")
print(f"Размер: {submission.shape}")
print("\nПервые 5 строк:")
print(submission.head())

# 5. Проверка формата
print("\nПроверка формата...")
sample_submission = pd.read_csv("../data/sample_submission.csv")
print("sample_submission columns:", sample_submission.columns.tolist())
print("Наш submission columns:", submission.columns.tolist())

if submission.columns.tolist() == sample_submission.columns.tolist():
    print("✅ Формат правильный!")
else:
    print("❌ Формат неправильный!")
    print("Исправь порядок колонок")