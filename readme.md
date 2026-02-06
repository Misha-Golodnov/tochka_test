# Классификатор качества текстов

Решение задачи классификации текстов по трем критериям для фильтрации датасетов претрейна LLM.

## Установка

bash

git clone https://github.com/Misha-Golodnov/tochka_test
cd tochka_test
pip install -r requirements.txt

## Быстрый старт

pip install -r requirements.txt

# Создайте папку для моделей
mkdir -p models

# Запустите обучение (займет 1-3 часа на CPU)
python scripts/train.py

python scripts/predict.py
# Или используйте альтернативный скрипт:
python create_submission.py

## Методология

Архитектура решения
3 отдельных классификатора на основе rubert-tiny2

Fine-tuning предобученной BERT-модели

Обогащение данных: текст + reasoning при обучении

Игнорирование примеров с меткой 0.5 (согласно условию задачи)

## Технологический стек

PyTorch - фреймворк для нейронных сетей

Transformers (Hugging Face) - BERT-модели и токенизация

Pandas - обработка табличных данных

Scikit-learn - метрики и валидация

PyArrow - чтение Parquet файлов

# Основные библиотеки

PyTorch - фреймворк для нейронных сетей

Transformers (Hugging Face) - BERT-модели и токенизация

Pandas - обработка табличных данных

Scikit-learn - метрики и валидация

PyArrow - чтение Parquet файлов

# Модель

rubert-tiny2 - легкая русская BERT-модель (47МБ)

## Описание файлов

# Файлы данных
train.parquet - тренировочные данные: 15,000 текстов с разметкой по 3 критериям

test.parquet - тестовые данные: 5,000 текстов для предсказания

sample_submission.csv - пример формата файла с ответами

# Исходный код
data_preprocessing.py - обработка и подготовка данных

model_training.py - обучение моделей

prediction.py - создание предсказаний

# Скрипты
train.py - Обучает 3 модели (по одной на каждый критерий)

predict.py - Создает предсказания на тестовых данных


