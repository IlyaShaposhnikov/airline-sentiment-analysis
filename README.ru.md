**[English Version / На английском](README.md)**

# Анализ тональности отзывов об авиакомпаниях в сети Twitter (X)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CI Status](https://github.com/IlyaShaposhnikov/airline-sentiment-analysis/actions/workflows/test.yml/badge.svg)](https://github.com/IlyaShaposhnikov/airline-sentiment-analysis/actions)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Сквозной ML/NLP-пайплайн для предсказания тональности твитов об авиакомпаниях. Включает обучение с учетом уверенности разметчиков (confidence-aware), объяснимый ИИ, готовый к использованию в рабочей среде REST API, интерактивную панель управления и CI с комплексным тестированием.

## Обзор

Проект демонстрирует полный жизненный цикл машинного обучения: от загрузки и предобработки данных до обучения модели, оценки, интерпретации и промышленного развертывания. Построен с использованием современных практик Python, модульной архитектуры и строгого тестирования.

- **Датасет**: [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) (с указанием уверенности аннотаций)
- **Модель**: Логистическая регрессия с TF-IDF векторизацией и взвешиванием по уверенности аннотаций
- **Интерпретируемость**: Веса модели + объяснения через SHAP (SHapley Additive exPlanations) — метод, который показывает вклад каждого слова в предсказание тональности на основе значений Шепли (опционально)
- **Развертывание**: FastAPI REST API + интерактивная панель на Streamlit
- **Качество**: Покрытие тестами >90% для основного ML-кода и сервисной логики, CI через GitHub Actions, аудит безопасности

## Возможности

| Категория | Ключевые особенности |
|-----------|---------------------|
| **Данные и предобработка** | Фильтрация по порогу уверенности, настраиваемая очистка (URL, упоминания, пунктуация), опциональная лемматизация, TF-IDF/Count векторизация |
| **Обучение и оценка** | Обучение с весами по уверенности, стратифицированное разделение, комплексные метрики (F1, ROC-AUC, матрица ошибок), авто-экспорт в JSON/CSV |
| **Интерпретируемость** | Топ-слова с наибольшим вкладом по классам, объяснения для отдельных предсказаний, при отсутствии пакета SHAP используется резервный метод |
| **API и сервис** | Асинхронно-безопасные эндпоинты FastAPI, валидация через Pydantic v2, потокобезопасный сервис модели, CORS, обработка таймаутов |
| **Панель управления** | Интерфейс для одиночных и пакетных предсказаний, экспорт в CSV/JSON, проверка здоровья API в реальном времени, сохранение состояния сессии |
| **Тестирование и CI** | Юнит- и интеграционные тесты, условные воркфлоу для PR/main, pytest-timeout, проверка уязвимостей через pip-audit, интеграция с Codecov |

## Архитектура

```
airline-sentiment-analysis/
├── configs/config.yaml          # Централизованная конфигурация (вложенная, валидируемая)
├── data/
│   ├── Tweets.csv               # Исходный датасет (загружается пользователем)
│   └── README.md                # Документация датасета
├── src/
│   ├── constants.py             # Глобальные константы: маппинг целевых меток, пути, значения по умолчанию
│   ├── data_loader.py           # Загрузка данных и фильтрация по уверенности
│   ├── preprocessing.py         # Очистка текста, лемматизация, векторизация
│   ├── models.py                # Обучение, оценка, сохранение моделей
│   ├── metrics.py               # Вычисление метрик и отчетность
│   ├── interpretability.py      # Объяснения через веса модели / SHAP
│   ├── api/                     # Сервисный слой FastAPI
│   │   ├── config.py            # Настройки API и переопределения через env-переменные
│   │   ├── models.py            # Схемы запросов/ответов Pydantic
│   │   ├── services.py          # Потокобезопасный ModelService
│   │   └── main.py              # Точка входа FastAPI и маршруты
│   ├── utils/logging_config.py  # Централизованная настройка логирования
│   └── dashboard.py             # Веб-интерфейс на Streamlit
├── scripts/
│   ├── train.py                 # CLI-пайплайн обучения
│   └── predict.py               # CLI-инференс и пакетный экспорт
├── tests/
│   ├── conftest.py              # Общие фикстуры pytest, маркеры, глобальная конфигурация тестов
│   ├── test_preprocessing.py    # Юнит-тесты очистки текста, токенизации, векторизации
│   ├── test_ml_models.py        # Юнит-тесты обучения, оценки, сохранения моделей
│   ├── test_api_models.py       # Юнит-тесты схем Pydantic: валидация, сериализация, ограничения
│   ├── test_api_services.py     # Юнит-тесты ModelService: асинхронность, потокобезопасность, обработка ошибок
│   └── test_integration.py      # Сквозные тесты пайплайна: config → data → model → API
├── pytest.ini                   # Конфигурация pytest: маркеры, фильтры, режим asyncio, опции по умолчанию
├── .github/workflows/test.yml   # CI-пайплайн
├── .gitignore                   # Шаблоны игнорирования для Git
├── README.md                    # Документация проекта (английский)
├── README.ru.md                 # Документация проекта (русский)
└── requirements.txt             # Зависимости проекта
```

## Быстрый старт

### 1. Настройка окружения
```bash
git clone https://github.com/IlyaShaposhnikov/airline-sentiment-analysis.git
cd airline-sentiment-analysis

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Подготовка данных
Скачайте `Tweets.csv` с [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) и поместите в директорию `data/`:
```
data/
└── Tweets.csv
```

### 3. Обучение модели
```bash
python scripts/train.py
# Опциональные флаги:
# --binary-mode            # Обучать только на 2 классах: positive/negative
# --no-plots               # Пропустить генерацию визуализаций
# --explain --n-explain 5  # Сгенерировать объяснения предсказаний
```
Артефакты будут сохранены в `artifacts/` (пакет модели, метрики, графики, логи).

### 4. Запуск REST API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Интерактивная документация: http://localhost:8000/docs

### 5. Запуск панели управления
```bash
streamlit run src/dashboard.py
```
Интерфейс откроется по адресу: http://localhost:8501

## Конфигурация

Все настройки задаются в `configs/config.yaml`. Ключевые секции:

```yaml
data:
  confidence_threshold: 0.7  # Фильтровать аннотации с низкой уверенностью

preprocessing:
  vectorizer: { type: "tfidf", max_features: 2000, ngram_range: [1, 2] }
  cleaning: { remove_urls: true, remove_mentions: false, remove_special_chars: true }

model:
  training: { max_iter: 500, class_weight: "balanced", use_confidence_weights: true }
  regularization: { solver: "lbfgs", penalty: "l2", C: 1.0 }

serving:
  api: { enabled: true, port: 8000, limits: { max_text_length: 1000, max_batch_size: 100 } }
  dashboard: { enabled: false, title: "Airline Sentiment Predictor" }
```
> 💡 Переменные окружения могут переопределять любое значение конфига (например, `MODEL_PATH`, `API_PORT`).

## Тестирование и CI

### Локальные тесты
```bash
# Запустить полный пакет тестов
pytest tests/ -v --cov=src --cov-report=term-missing

# Быстрый режим для PR (пропустить интеграционные тесты)
pytest tests/ -v -m "not integration"
```

### CI-пайплайн
Автоматически запускается при `push` и `pull_request`:
- Тесты на Python 3.10, 3.11, 3.12
- Быстрый режим для PR, полный режим для `main`/`develop`
- Сканирование уязвимостей через `pip-audit`
- Покрытие загружается в Codecov + артефакты GitHub
- Защита от зависаний и отмена дублирующих запусков

## Использование API

```bash
# Одиночное предсказание
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great flight, excellent service!", "explain": true}'

# Пакетное предсказание
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing!", "Terrible delay", "Meh"], "explain": false}'
```

---

## Зависимости

| Категория | Пакеты |
|-----------|--------|
| Основные | `numpy`, `pandas`, `scikit-learn`, `scipy` |
| NLP | `nltk`, `shap` (опционально) |
| API | `fastapi`, `uvicorn`, `pydantic>=2` |
| Интерфейс | `streamlit`, `matplotlib`, `seaborn` |
| Тестирование | `pytest`, `pytest-cov`, `pytest-asyncio`, `pytest-timeout`, `pip-audit` |

## Автор

Илья Шапошников | [E-mail](mailto:ilia.a.shaposhnikov@gmail.com) | [LinkedIn](https://linkedin.com/in/iliashaposhnikov)

**[English Version / На английском](README.md)**