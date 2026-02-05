import pandas as pd
import numpy as np

# Загрузим train данные
train = pd.read_parquet("data/train.parquet")
test = pd.read_parquet("data/test.parquet")

print("Train данные")
print(f"Размер: {train.shape}")
print("Колонки:", train.columns.tolist())
print("\nПример данных:")
print(train.head())

print("\n=== Test данные ===")
print(f"Размер: {test.shape}")
print("Колонки:", test.columns.tolist())

print("\n=== Распределение меток в train ===")
for col in ['integrity', 'factuality', 'truthfulness']:
    print(f"\n{col}:")
    print(train[col].value_counts(normalize=True))