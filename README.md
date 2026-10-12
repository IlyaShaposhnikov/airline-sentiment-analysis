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
- **Models**: TF-IDF or sentence embeddings (Sentence-Transformers) × Logistic Regression or a PyTorch MLP, all with confidence-aware sample weighting; a reproducible multi-seed comparison is included
- **Interpretability**: Model weights + SHAP explanations (optional)
- **Deployment**: FastAPI REST API + Streamlit interactive dashboard
- **Quality**: >90% test coverage for core ML and service logic, GitHub Actions CI, security auditing

## Features

| Category | Highlights |
|----------|------------|
| **Data & Preprocessing** | Confidence threshold filtering, configurable cleaning (URLs, mentions, punctuation), optional lemmatization, TF-IDF/Count vectorization; cleaning is embedded in the vectorizer, so training and inference process text identically |
| **Training & Evaluation** | Confidence-weighted learning, stratified splits, comprehensive metrics (F1, ROC-AUC, confusion matrix), auto-export to JSON/CSV, multi-seed comparison of feature extractors × classifiers |
| **Interpretability** | Top contributing words per class, per-prediction explanations, SHAP fallback if package unavailable |
| **API & Serving** | Async-safe FastAPI endpoints, Pydantic v2 validation, thread-safe model service, CORS, timeout handling |
| **Dashboard** | Single/batch prediction UI, CSV/JSON export, real-time API health check, session state persistence |
| **Testing & CI** | Unit + integration tests, conditional PR/main workflows, pytest-timeout, pip-audit security checks, Codecov integration |

## Model Performance & Results

### Approach Comparison

Two feature extractors × two classifiers, evaluated by `scripts/compare.py` on **5 stratified splits** (seeds 42–46). For each seed all four configurations share the same split and confidence weights.

| Configuration | Accuracy | F1 (macro) | ROC-AUC | Latency, ms/text<br>(batch / single) |
|---|---|---|---|---|
| **3 classes** | | | | |
| TF-IDF + LogReg | 0.843 ± 0.008 | 0.785 ± 0.011 | 0.938 ± 0.006 | 0.04 / 0.4 |
| TF-IDF + MLP | 0.845 ± 0.007 | 0.783 ± 0.010 | 0.938 ± 0.005 | 0.05 / 1.0 |
| Embeddings + LogReg | **0.854 ± 0.004** | **0.805 ± 0.007** | 0.955 ± 0.005 | 5.5 / 14.9 |
| Embeddings + MLP | 0.853 ± 0.006 | 0.804 ± 0.008 | **0.956 ± 0.004** | 5.3 / 17.2 |
| **Binary (positive / negative)** | | | | |
| TF-IDF + LogReg | 0.934 ± 0.005 | 0.888 ± 0.008 | 0.969 ± 0.006 | 0.04 / 0.3 |
| TF-IDF + MLP | 0.924 ± 0.004 | 0.875 ± 0.006 | 0.969 ± 0.003 | 0.07 / 0.8 |
| Embeddings + LogReg | **0.949 ± 0.004** | **0.916 ± 0.006** | **0.986 ± 0.002** | 5.6 / 15.7 |
| Embeddings + MLP | **0.949 ± 0.007** | **0.916 ± 0.010** | **0.986 ± 0.002** | 5.0 / 14.9 |

*Mean ± std over seeds. TF-IDF: 2,000 uni/bigrams fitted on the training split; embeddings: `all-MiniLM-L6-v2` (384-d, frozen); MLP: one hidden layer of 128 with early stopping. Latency is end-to-end from raw text on a laptop CPU.*

**Takeaways**

- **The text representation matters more than the classifier.** Sentence embeddings improve every metric over TF-IDF with either classifier: +2.0 pp macro-F1 for 3 classes, +2.8 pp for binary, +0.017 ROC-AUC in both — two to three standard deviations, consistently across seeds.
- **A non-linear classifier adds nothing here.** The MLP matches logistic regression on embeddings and is no better (binary: worse) on TF-IDF. With good features, a linear model is enough.
- **Quality has a price.** Embeddings are ~40× slower per single request (≈15 ms vs 0.4 ms) and over 100× in batch, need torch plus a 90 MB model, and give up word-level explanations.
- **Practical choice:** TF-IDF + LogReg stays the default — fast, lightweight and explainable. Embeddings + LogReg is the pick when accuracy matters more than latency and interpretability.

Reproduce: `python scripts/compare.py` (and `--binary-mode`). Embeddings are encoded once and cached.

### Baseline Model Details (TF-IDF + LogReg)

| | Multiclass (3 classes) | Binary (positive/negative) |
|--------|----------------------|--------------------------|
| **Confusion Matrix** | ![Multiclass CM](docs/images/multiclass/confusion_matrix.png) | ![Binary CM](docs/images/binary/confusion_matrix.png) |
| **Positive Features** | ![Positive Multi](docs/images/multiclass/feature_importance_positive.png) | ![Positive Binary](docs/images/binary/feature_importance_positive.png) |
| **Negative Features** | ![Negative Multi](docs/images/multiclass/feature_importance_negative.png) | ![Negative Binary](docs/images/binary/feature_importance_negative.png) |
| **Neutral Features** | ![Neutral Multi](docs/images/multiclass/feature_importance_neutral.png) | *N/A* |

> 💡 **Notes**:
> - Plots for the `random_state=42` split (25% held-out test set) of 10,768 tweets with annotator confidence ≥ 0.7; metrics for this model are in the TF-IDF + LogReg rows above
> - Binary mode excludes neutral class (target=2), focusing on clear positive/negative signals
> - Feature importance shows top 20 words by logistic regression coefficient weight
> - Confusion matrices are normalized (row-wise) for fair class comparison

## Architecture

```
airline-sentiment-analysis/
├── configs/
│   ├── config.yaml              # Centralized configuration (nested, validated)
│   └── CONFIG_GUIDE.md          # Detailed reference for config parameters, validation rules, and env overrides
├── data/
│   ├── Tweets.csv               # Raw dataset (downloaded by user)
│   └── DATA_GUIDE.md            # Dataset documentation
├── docs/images/                 # Model performance visualizations (multiclass/binary)
├── src/
│   ├── constants.py             # Project-wide constants: target mappings, paths, default values
│   ├── data_loader.py           # Data ingestion & confidence filtering
│   ├── preprocessing.py         # Text cleaning, lemmatization, vectorization
│   ├── embeddings.py            # Sentence-Transformers embeddings as a sklearn-style vectorizer (optional)
│   ├── models.py                # Training, evaluation, persistence
│   ├── torch_models.py          # PyTorch MLP classifier with a sklearn API (optional)
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
│   ├── predict.py               # CLI inference & batch export
│   └── compare.py               # TF-IDF/embeddings x LogReg/MLP comparison over several seeds
├── tests/
│   ├── conftest.py              # Shared pytest fixtures, markers, and global test configuration
│   ├── test_torch_models.py     # MLP training, early stopping, persistence, pipeline & API
│   ├── test_shap_lazy_import.py # shap is imported only when SHAP explanations are requested
│   ├── test_api_model_info.py   # Model metadata in ModelService and /health
│   ├── test_compare.py          # Comparison script: outputs, embedding cache, binary mode
│   ├── test_metrics.py          # Metric definitions (macro F1 in binary mode)
│   ├── test_runtime.py          # Windows torch/scikit-learn OpenMP workaround and import order
│   ├── test_embeddings.py       # Embedding vectorizer, pipeline & API path (fake encoder, no download)
│   ├── test_data_loader.py      # Unit tests for the train/test split helper (alignment, stratification)
│   ├── test_preprocessing.py    # Unit tests for text cleaning, tokenization, and vectorization
│   ├── test_ml_models.py        # Unit tests for model training, evaluation, and persistence logic
│   ├── test_api_models.py       # Unit tests for Pydantic schemas: validation, serialization, constraints
│   ├── test_api_services.py     # Unit tests for ModelService: async handling, thread safety, error propagation
│   ├── test_integration.py      # End-to-end pipeline tests: config → data → model → API
│   └── test_train_pipeline.py   # Regression tests for scripts/train.py artifacts (no leakage, aligned reports)
├── pytest.ini                   # Pytest configuration: markers, filters, asyncio mode, default options
├── .github/workflows/test.yml   # CI pipeline
├── .gitignore                   # Git ignore patterns
├── README.md                    # Project documentation (English)
├── README.ru.md                 # Project documentation (Russian)
├── requirements.txt             # Project dependencies
└── requirements-dl.txt          # Optional deep-learning dependencies (torch, sentence-transformers)
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

**Optional: sentence embeddings instead of TF-IDF.** Texts are encoded with a pretrained [Sentence-Transformers](https://www.sbert.net/) model (`all-MiniLM-L6-v2` by default, see `preprocessing.embedding` in the config) and classified by the same logistic regression:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only build
pip install -r requirements-dl.txt
python scripts/train.py --vectorizer sentence_embedding --output-dir artifacts/embedding_lr
```
The model is downloaded once (~90 MB) into the Hugging Face cache.

**Optional: PyTorch MLP instead of logistic regression** (works with either feature type; settings in `model.mlp`, early stopping on a validation slice of the training split):
```bash
python scripts/train.py --model mlp --output-dir artifacts/tfidf_mlp
python scripts/train.py --model mlp --vectorizer sentence_embedding --output-dir artifacts/embedding_mlp
```
Word-level explanations and feature-importance plots require TF-IDF features **and** a linear model; for embeddings or the MLP the API returns predictions with `explanation: null`. To serve another model, point the API to its bundle: `MODEL_PATH=artifacts/embedding_mlp/model_bundle.joblib uvicorn src.api.main:app`.

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
  cleaning: { remove_urls: true, remove_mentions: true, remove_special_chars: true }

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

# Health check: also reports which model is served and whether it can explain
curl http://localhost:8000/health
# {"status": "healthy", "model_loaded": true, "model_type": "LogisticRegression",
#  "vectorizer_type": "TfidfVectorizer", "explanations_supported": true, ...}
```
Models without word-level explanations (sentence embeddings, MLP) return `"explanation": null` even with `"explain": true`; the dashboard disables the explanation checkbox for them.

## Troubleshooting

**Windows: `Windows fatal exception: access violation` while encoding sentence embeddings.**
A rare, non-deterministic native crash inside torch (observed with torch 2.14 CPU on Windows). The project already applies a mitigation (see `src/__init__.py`). If it still happens, rerun the command, or run with a single OpenMP thread (reliable, ~2x slower encoding):
```bash
OMP_NUM_THREADS=1 python scripts/train.py --vectorizer sentence_embedding
```

## Dependencies

| Category | Packages |
|----------|----------|
| Core | `numpy`, `pandas`, `scikit-learn`, `scipy` |
| NLP | `nltk`, `shap` (optional) |
| API | `fastapi`, `uvicorn`, `pydantic>=2` |
| UI | `streamlit`, `matplotlib`, `seaborn` |
| Deep learning (optional) | `torch`, `sentence-transformers` |
| Testing | `pytest`, `pytest-cov`, `pytest-asyncio`, `pytest-timeout`, `pip-audit` |

## Author

Ilya Shaposhnikov | [E-mail](mailto:ilia.a.shaposhnikov@gmail.com) | [LinkedIn](https://linkedin.com/in/iliashaposhnikov)

**[Russian Version / На русском](README.ru.md)**