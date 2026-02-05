#!/usr/bin/env python3
"""Скрипт для обучения моделей"""

import sys
import os
from pathlib import Path

# ПРАВИЛЬНЫЙ ПУТЬ К ИМПОРТУ src
# Добавляем родительскую директорию в путь Python
current_dir = Path(__file__).parent
project_root = current_dir.parent  # Поднимаемся на уровень выше scripts/
sys.path.insert(0, str(project_root))

print(f"Python path: {sys.path[:3]}...")  # Для отладки

try:
    # Теперь импортируем из src
    from src.data_preprocessing import load_and_prepare_data
    from src.model_training import train_all_models

    print("Модули успешно импортированы")
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print(f"Текущая директория: {os.getcwd()}")
    print(f"Содержимое src/: {os.listdir('src') if os.path.exists('src') else 'папка не существует'}")
    sys.exit(1)


def main():
    # Пути к данным - используем абсолютные пути
    project_root = Path(__file__).parent.parent
    train_path = project_root / "data" / "train.parquet"
    output_dir = project_root / "models"

    print(f"Проект расположен в: {project_root}")
    print(f"Train данные: {train_path}")
    print(f"Выходная директория: {output_dir}")

    # Проверка существования файлов
    if not train_path.exists():
        print(f"Файл не найден: {train_path}")
        print(f"Содержимое data/: {list((project_root / 'data').iterdir())}")
        sys.exit(1)

    print("\nЗагрузка данных")
    train_data = load_and_prepare_data(str(train_path))

    print("\nОбучение моделей...")
    models, tokenizers = train_all_models(
        train_data=train_data,
        model_name='cointegrated/rubert-tiny2',
        output_dir=str(output_dir)
    )

    print("\n Обучение завершено!")
    print(f"Модели сохранены в {output_dir}")


if __name__ == "__main__":
    main()