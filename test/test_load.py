import sys
import os
sys.path.append('..')

# Проверка загрузки данных
print("Тест загрузки данных...")
try:
    import pandas as pd
    train = pd.read_parquet("../data/train.parquet")
    test = pd.read_parquet("../data/test.parquet")
    print(f"✓ Train загружен: {train.shape}")
    print(f"✓ Test загружен: {test.shape}")
    print(f"Колонки train: {train.columns.tolist()}")
    print(f"Колонки test: {test.columns.tolist()}")
except Exception as e:
    print(f"✗ Ошибка: {e}")