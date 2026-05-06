# Airline Sentiment Analysis

Sentiment classification of airline tweets using Logistic Regression, with confidence-aware training and interpretability tools.

## Features
- Multiclass & binary sentiment classification (positive/negative/neutral)
- Confidence-aware training using `airline_sentiment_confidence` and `negativereason_confidence`
- TF-IDF / Count vectorization with configurable preprocessing
- Model interpretation: weight analysis + optional SHAP support
- Simple FastAPI endpoint (`/predict`) and Streamlit dashboard (optional)

## Quick Start

### 1. Clone & install
```bash
git clone https://github.com/IlyaShaposhnikov/airline-sentiment-analysis.git
cd airline-sentiment-analysis
pip install -r requirements.txt
```

### 2. Prepare data
Download `Tweets.csv` from [Kaggle: Twitter Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) and place it in `data/`.

### 3. Train model
```bash
python scripts/train.py --config configs/config.yaml
```

### 4. Run API (optional)
```bash
uvicorn src.api:app --reload
# Then: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "Great flight!"}'
```

### 5. Launch Dashboard (optional)
```bash
streamlit run src/dashboard.py
```

## Configuration
Edit `configs/config.yaml` to adjust:
- `confidence_threshold`: filter low-confidence labels (default: 0.7)
- `vectorizer`: TF-IDF or Count, max_features, ngram_range
- `model`: LogisticRegression params, class_weight, use_confidence_weights

## Evaluation Metrics
- Accuracy, F1-score (macro/weighted), ROC-AUC (OvO for multiclass)
- Confusion matrix visualization
- Per-airline performance breakdown (optional)

