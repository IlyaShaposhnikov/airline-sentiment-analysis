**[Russian Version / На русском](README.ru.md)**

# Airline Sentiment Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CI Status](https://github.com/IlyaShaposhnikov/airline-sentiment-analysis/actions/workflows/test.yml/badge.svg)](https://github.com/IlyaShaposhnikov/airline-sentiment-analysis/actions)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

End-to-end ML/NLP pipeline for predicting airline tweet sentiment. Features confidence-aware training, explainable AI, production-ready REST API, interactive dashboard, and CI with comprehensive testing.

## Overview

This project demonstrates a complete machine learning lifecycle: from raw data ingestion and preprocessing to model training, evaluation, interpretation, and production deployment. Built with modern Python practices, modular architecture, and rigorous testing.

- **Dataset**: [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) (confidence-filtered)
- **Model**: Logistic Regression with TF-IDF vectorization & confidence-aware sample weighting
- **Interpretability**: Model weights + SHAP explanations (optional)
- **Deployment**: FastAPI REST API + Streamlit interactive dashboard
- **Quality**: >90% test coverage for core ML and service logic, GitHub Actions CI, security auditing

## Features

| Category | Highlights |
|----------|------------|
| **Data & Preprocessing** | Confidence threshold filtering, configurable cleaning (URLs, mentions, punctuation), optional lemmatization, TF-IDF/Count vectorization |
| **Training & Evaluation** | Confidence-weighted learning, stratified splits, comprehensive metrics (F1, ROC-AUC, confusion matrix), auto-export to JSON/CSV |
| **Interpretability** | Top contributing words per class, per-prediction explanations, SHAP fallback if package unavailable |
| **API & Serving** | Async-safe FastAPI endpoints, Pydantic v2 validation, thread-safe model service, CORS, timeout handling |
| **Dashboard** | Single/batch prediction UI, CSV/JSON export, real-time API health check, session state persistence |
| **Testing & CI** | Unit + integration tests, conditional PR/main workflows, pytest-timeout, pip-audit security checks, Codecov integration |

## Architecture

```
airline-sentiment-analysis/
├── configs/config.yaml
│   ├── config.yaml              # Centralized configuration (nested, validated)
│   └── CONFIG_GUIDE.md          # Detailed reference for config parameters, validation rules, and env overrides
├── data/
│   ├── Tweets.csv               # Raw dataset (downloaded by user)
│   └── DATA_GUIDE.md            # Dataset documentation
├── src/
│   ├── constants.py             # Project-wide constants: target mappings, paths, default values
│   ├── data_loader.py           # Data ingestion & confidence filtering
│   ├── preprocessing.py         # Text cleaning, lemmatization, vectorization
│   ├── models.py                # Training, evaluation, persistence
│   ├── metrics.py               # Metrics computation & reporting
│   ├── interpretability.py      # Weight/SHAP explanations
│   ├── api/                     # FastAPI service layer
│   │   ├── config.py            # API settings & env overrides
│   │   ├── models.py            # Pydantic request/response schemas
│   │   ├── services.py          # Thread-safe ModelService
│   │   ├── main.py              # FastAPI entry point & routes
│   │   └── API_REFERENCE.md     # Technical reference for REST endpoints, Pydantic schemas, and error handling
│   ├── utils/logging_config.py  # Centralized logging setup: formatters, handlers, log levels
│   └── dashboard.py             # Streamlit web UI
├── scripts/
│   ├── train.py                 # CLI training pipeline
│   └── predict.py               # CLI inference & batch export
├── tests/
│   ├── conftest.py              # Shared pytest fixtures, markers, and global test configuration
│   ├── test_preprocessing.py    # Unit tests for text cleaning, tokenization, and vectorization
│   ├── test_ml_models.py        # Unit tests for model training, evaluation, and persistence logic
│   ├── test_api_models.py       # Unit tests for Pydantic schemas: validation, serialization, constraints
│   ├── test_api_services.py     # Unit tests for ModelService: async handling, thread safety, error propagation
│   └── test_integration.py      # End-to-end pipeline tests: config → data → model → API
├── pytest.ini                   # Pytest configuration: markers, filters, asyncio mode, default options
├── .github/workflows/test.yml   # CI pipeline
├── .gitignore                   # Git ignore patterns
├── README.md                    # Project documentation (English)
├── README.ru.md                 # Project documentation (Russian)
└── requirements.txt             # Project dependencies
```

## Quick Start

### 1. Setup Environment
```bash
git clone https://github.com/IlyaShaposhnikov/airline-sentiment-analysis.git
cd airline-sentiment-analysis

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Prepare Data
Download `Tweets.csv` from [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) and place it in the `data/` directory:
```
data/
└── Tweets.csv
```

### 3. Train Model
```bash
python scripts/train.py
# Optional flags:
# --binary-mode            # Train only positive/negative
# --no-plots               # Skip visualization generation
# --explain --n-explain 5  # Generate prediction explanations
```
Artifacts will be saved to `artifacts/` (model bundle, metrics, plots, logs).

### 4. Run REST API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive docs: http://localhost:8000/docs

### 5. Launch Dashboard
```bash
streamlit run src/dashboard.py
```
UI opens at: http://localhost:8501

## Configuration

All behavior is controlled via `configs/config.yaml`. Key sections:

```yaml
data:
  confidence_threshold: 0.7  # Filter low-confidence annotations

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
> 💡 Environment variables can override any config value (e.g., `MODEL_PATH`, `API_PORT`).

## Testing & CI

### Local Tests
```bash
# Run full suite
pytest tests/ -v --cov=src --cov-report=term-missing

# Fast PR mode (skip integration tests)
pytest tests/ -v -m "not integration"
```

### CI Pipeline
Automatically runs on `push` and `pull_request`:
- Tests on Python 3.10, 3.11, 3.12
- Fast mode for PRs, full mode for `main`/`develop`
- `pip-audit` security scanning
- Coverage uploaded to Codecov + GitHub artifacts
- Timeout protection & duplicate run cancellation

## API Usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great flight, excellent service!", "explain": true}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing!", "Terrible delay", "Meh"], "explain": false}'
```

## Dependencies

| Category | Packages |
|----------|----------|
| Core | `numpy`, `pandas`, `scikit-learn`, `scipy` |
| NLP | `nltk`, `shap` (optional) |
| API | `fastapi`, `uvicorn`, `pydantic>=2` |
| UI | `streamlit`, `matplotlib`, `seaborn` |
| Testing | `pytest`, `pytest-cov`, `pytest-asyncio`, `pytest-timeout`, `pip-audit` |

## Author

Ilya Shaposhnikov | [E-mail](mailto:ilia.a.shaposhnikov@gmail.com) | [LinkedIn](https://linkedin.com/in/iliashaposhnikov)

**[Russian Version / На русском](README.ru.md)**