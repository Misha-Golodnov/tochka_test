import os
import pandas as pd

print("=" * 60)
print("ИТОГОВАЯ ПРОВЕРКА ПРОЕКТА")
print("=" * 60)

checks_passed = 0
total_checks = 8

# 1. Проверка структуры папок
print("\n1. Проверка структуры папок...")
required_dirs = ['data', 'src', 'scripts']
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"   ✓ {dir_name}/ существует")
        checks_passed += 1
    else:
        print(f"   ✗ {dir_name}/ не существует")

# 2. Проверка файлов данных
print("\n2. Проверка файлов данных...")
required_files = [
    'data/train.parquet',
    'data/test.parquet',
    'data/sample_submission.csv'
]
for file_path in required_files:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024 / 1024
        print(f"   ✓ {file_path} ({size:.1f} MB)")
        checks_passed += 1
    else:
        print(f"   ✗ {file_path} не найден")

# 3. Проверка исходного кода
print("\n3. Проверка исходного кода...")
src_files = ['data_preprocessing.py', 'model_training.py', 'prediction.py']
for file_name in src_files:
    file_path = f"src/{file_name}"
    if os.path.exists(file_path):
        print(f"   ✓ {file_path}")
        checks_passed += 1
    else:
        print(f"   ✗ {file_path} не найден")

# 4. Проверка скриптов
print("\n4. Проверка скриптов...")
script_files = ['train.py', 'predict.py']
for file_name in script_files:
    file_path = f"scripts/{file_name}"
    if os.path.exists(file_path):
        print(f"   ✓ {file_path}")
        checks_passed += 1
    else:
        print(f"   ✗ {file_path} не найден")

# 5. Проверка зависимостей
print("\n5. Проверка requirements.txt...")
if os.path.exists("../requirements.txt"):
    with open("../requirements.txt", 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"   ✓ requirements.txt ({len(lines)} зависимостей)")
    checks_passed += 1
else:
    print("   ✗ requirements.txt не найден")

# 6. Проверка .gitignore
print("\n6. Проверка .gitignore...")
if os.path.exists(".gitignore"):
    print("   ✓ .gitignore существует")
    checks_passed += 1
else:
    print("   ✗ .gitignore не найден")

# 7. Проверка README
print("\n7. Проверка README.md...")
if os.path.exists("../readme.md"):
    print("   ✓ README.md существует")
    checks_passed += 1
else:
    print("   ✗ README.md не найден")

# 8. Проверка submission файла
print("\n8. Проверка submission файла...")
try:
    submission = pd.read_csv("submission_sample.csv")
    if len(submission) == 20:  # 20 примеров в test_sample
        print(f"   ✓ submission_sample.csv ({len(submission)} строк)")
        checks_passed += 1
    else:
        print(f"   ✗ Неправильный размер: {len(submission)} строк")
except:
    print("   ✗ submission_sample.csv не найден или ошибка чтения")

# ИТОГ
print("\n" + "=" * 60)
print(f"ПРОВЕРКА ЗАВЕРШЕНА")
print(f"Пройдено: {checks_passed}/{total_checks}")
print("=" * 60)

if checks_passed == total_checks:
    print("🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ! МОЖЕШЬ ЗАПУСКАТЬ НА ПОЛНЫХ ДАННЫХ!")
    print("\nСледующие шаги:")
    print("1. Запусти полное обучение: python scripts/train.py")
    print("2. Создай предсказания: python scripts/predict.py")
    print("3. Загрузи на GitHub")
else:
    print("⚠️ Есть проблемы. Исправь отмеченные ошибки.")