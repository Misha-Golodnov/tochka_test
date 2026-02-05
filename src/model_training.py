import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import os
import transformers  # Для проверки версии


# Определяем Dataset
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


def compute_metrics(eval_pred):
    """Вычисление метрик для оценки модели"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    f1 = f1_score(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)

    return {
        'f1': f1,
        'accuracy': acc
    }


def train_model_for_task(task_name, train_texts, train_labels, val_texts, val_labels,
                         model_name='cointegrated/rubert-tiny2', output_dir='models'):
    """Обучение модели для одной задачи"""
    print(f"Обучение модели для задачи: {task_name}")

    # Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Создание датасетов
    train_dataset = TextQualityDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextQualityDataset(val_texts, val_labels, tokenizer)

    # Модель
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # Проверяем версию transformers для правильных аргументов
    print(f"Версия transformers: {transformers.__version__}")

    # Создаем аргументы обучения в зависимости от версии
    if hasattr(TrainingArguments, 'eval_strategy'):
        # Новая версия (>= 4.30.0)
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, task_name),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=100,  # Уменьшил для быстрого старта
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            eval_strategy='epoch',  # НОВОЕ ИМЯ
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            save_total_limit=2,
            seed=42,
            report_to="none",  # Отключаем отчеты
            disable_tqdm=False  # Включаем прогресс-бар
        )
    else:
        # Старая версия (< 4.30.0)
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, task_name),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy='epoch',  # СТАРОЕ ИМЯ
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            save_total_limit=2,
            seed=42,
            report_to="none",
            disable_tqdm=False
        )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Обучение
    print(f"Начало обучения...")
    trainer.train()

    # Сохранение лучшей модели
    best_model_path = os.path.join(output_dir, f"{task_name}_best")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    # Оценка на валидации
    eval_results = trainer.evaluate()
    print(f"\nРезультаты для {task_name}:")
    print(f"F1-score: {eval_results['eval_f1']:.4f}")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")

    return trainer.model, tokenizer


def train_all_models(train_data, model_name='cointegrated/rubert-tiny2', output_dir='models'):
    """Обучение моделей для всех трех задач"""

    os.makedirs(output_dir, exist_ok=True)

    models = {}
    tokenizers = {}

    for task in ['integrity', 'factuality', 'truthfulness']:
        task_data = train_data[task]

        print(f"\n Начинаю обучение для {task}...")
        model, tokenizer = train_model_for_task(
            task_name=task,
            train_texts=task_data['train_texts'],
            train_labels=task_data['train_labels'],
            val_texts=task_data['val_texts'],
            val_labels=task_data['val_labels'],
            model_name=model_name,
            output_dir=output_dir
        )

        models[task] = model
        tokenizers[task] = tokenizer

    return models, tokenizers