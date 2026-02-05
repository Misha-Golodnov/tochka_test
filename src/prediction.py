import torch
import numpy as np
from tqdm import tqdm


def predict_batch(model, tokenizer, texts, device='cpu', batch_size=32):
    """Предсказание для батча текстов"""
    model.eval()
    model.to(device)

    predictions = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Предсказание"):
        batch_texts = texts[i:i + batch_size]

        # Токенизация батча
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )

        # Перемещение на устройство
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Предсказание
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_preds)

    return predictions


def create_submission(models, tokenizers, test_texts, test_uuids, device='cpu'):
    """Создание submission файла"""

    predictions = {}

    for task in ['integrity', 'factuality', 'truthfulness']:
        print(f"\nПредсказание для {task}...")

        # Базовый текст для предсказания (без reasoning в тесте)
        texts_for_prediction = [f"Текст: {text}" for text in test_texts]

        # Получаем предсказания
        task_predictions = predict_batch(
            model=models[task],
            tokenizer=tokenizers[task],
            texts=texts_for_prediction,
            device=device,
            batch_size=64
        )

        predictions[task] = task_predictions

    # Создаем DataFrame
    import pandas as pd
    submission_df = pd.DataFrame({
        'uuid': test_uuids,
        'integrity': predictions['integrity'],
        'truthfulness': predictions['truthfulness'],  # Внимание: именно такой порядок!
        'factuality': predictions['factuality']
    })

    return submission_df