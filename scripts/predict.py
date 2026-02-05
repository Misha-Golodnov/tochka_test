#!/usr/bin/env python3
"""Скрипт для создания предсказаний"""

import sys
import os
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# НАСТРОЙКА ПУТЕЙ
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

print(f"Корень проекта: {project_root}")

src_path = project_root / "src"
if not src_path.exists():
    print(f"Ошибка: Папка src не найдена: {src_path}")
    sys.exit(1)

print(f"Папка src найдена: {src_path}")


def load_test_data(test_path):
    """Простая функция загрузки тестовых данных"""
    print(f"Чтение тестовых данных из: {test_path}")
    test_df = pd.read_parquet(test_path)
    test_texts = test_df['text'].tolist()
    test_uuids = test_df.index.tolist()
    return test_texts, test_uuids


def create_submission(models, tokenizers, test_texts, test_uuids, device='cpu'):
    """Создание submission файла"""

    predictions = {}

    for task in ['integrity', 'factuality', 'truthfulness']:
        print(f"\nПредсказание для {task}...")

        model = models[task]
        tokenizer = tokenizers[task]
        model.eval()
        model.to(device)

        task_predictions = []
        batch_size = 32

        for i in tqdm(range(0, len(test_texts), batch_size), desc=f"{task}"):
            batch_texts = test_texts[i:i + batch_size]
            prepared_texts = [f"Текст: {text}" for text in batch_texts]

            encoding = tokenizer(
                prepared_texts,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask']
                )
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                task_predictions.extend(batch_preds)

        predictions[task] = task_predictions
        print(f"Предсказано значений: {len(task_predictions)}")

    submission_df = pd.DataFrame({
        'uuid': test_uuids,
        'integrity': predictions['integrity'],
        'truthfulness': predictions['truthfulness'],
        'factuality': predictions['factuality']
    })

    return submission_df


def load_models(models_dir='models'):
    """Загрузка обученных моделей"""

    models = {}
    tokenizers = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")

    for task in ['integrity', 'factuality', 'truthfulness']:
        model_path = os.path.join(models_dir, f"{task}_best")

        if not os.path.exists(model_path):
            print(f"Ошибка: Модель для {task} не найдена по пути: {model_path}")
            print(f"Проверьте что папка {models_dir} содержит подпапки:")
            print(f"  - integrity_best/")
            print(f"  - factuality_best/")
            print(f"  - truthfulness_best/")
            sys.exit(1)

        print(f"Загрузка модели для {task}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)

            models[task] = model
            tokenizers[task] = tokenizer
            print(f"  Модель загружена успешно")

        except Exception as e:
            print(f"  Ошибка загрузки модели: {e}")
            sys.exit(1)

    return models, tokenizers, device


def main():

    print("ЗАПУСК СКРИПТА ПРЕДСКАЗАНИЙ")


    # Пути к файлам
    test_path = project_root / "data" / "test.parquet"
    models_dir = project_root / "models"
    output_file = project_root / "submission.csv"

    print(f"Тестовые данные: {test_path}")
    print(f"Папка с моделями: {models_dir}")
    print(f"Выходной файл: {output_file}")

    # Проверка существования тестовых данных
    if not test_path.exists():
        print(f"Ошибка: Файл не найден: {test_path}")
        print("Убедитесь что файл test.parquet находится в папке data/")
        sys.exit(1)

    # Проверка существования моделей
    if not models_dir.exists():
        print(f"Ошибка: Папка с моделями не найдена: {models_dir}")
        print("Сначала запустите обучение моделей: python scripts/train.py")
        sys.exit(1)

    # Загрузка тестовых данных
    print("\nЗагрузка тестовых данных...")
    try:
        test_texts, test_uuids = load_test_data(test_path)
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        sys.exit(1)

    # Загрузка моделей
    print("\nЗагрузка обученных моделей...")
    models, tokenizers, device = load_models(str(models_dir))

    if len(models) != 3:
        print(f"Ошибка: Загружено только {len(models)} из 3 моделей")
        sys.exit(1)

    # Создание предсказаний
    print("\nСоздание предсказаний...")
    try:
        submission_df = create_submission(
            models=models,
            tokenizers=tokenizers,
            test_texts=test_texts,
            test_uuids=test_uuids,
            device=device
        )
    except Exception as e:
        print(f"Ошибка при создании предсказаний: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Сохранение результатов
    submission_df.to_csv(output_file, index=False)
    print(f"\nФайл submission.csv сохранен: {output_file}")
    print(f"Размер файла: {submission_df.shape}")

    # Проверка формата
    print("\nПроверка формата submission файла:")
    print(f"Колонки: {submission_df.columns.tolist()}")
    print(f"Количество строк: {len(submission_df)}")
    print(f"Первые 5 строк:")
    print(submission_df.head())

    # Проверяем что все значения 0 или 1
    print("\nПроверка значений:")
    for col in ['integrity', 'truthfulness', 'factuality']:
        unique_values = submission_df[col].unique()
        print(f"{col}: уникальные значения {sorted(unique_values)}")
        if set(unique_values) - {0, 1}:
            print(f"  Внимание: обнаружены значения кроме 0 и 1!")

    # Сравнение с sample_submission
    sample_path = project_root / "data" / "sample_submission.csv"
    if sample_path.exists():
        sample_df = pd.read_csv(sample_path)
        if list(submission_df.columns) == list(sample_df.columns):
            print("\nФормат колонок соответствует sample_submission.csv")
        else:
            print("\nВнимание: формат колонок не соответствует sample_submission.csv")
            print(f"Ожидалось: {sample_df.columns.tolist()}")
            print(f"Получено: {submission_df.columns.tolist()}")
            print("Порядок колонок важен: uuid, integrity, truthfulness, factuality")


    print("ПРЕДСКАЗАНИЯ УСПЕШНО СОЗДАНЫ")



if __name__ == "__main__":
    main()