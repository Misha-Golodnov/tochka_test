import sys
import os
sys.path.append('..')

print("Тест импорта модулей...")
try:
    from src.data_preprocessing import TextQualityDataset, load_and_prepare_data
    from src.model_training import train_all_models
    from src.prediction import create_submission
    print("✓ Все модули импортируются")
except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    print("Проверьте структуру проекта и пути импорта")