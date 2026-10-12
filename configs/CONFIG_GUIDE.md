# Configuration Guide

Comprehensive guide to configuring `configs/config.yaml` for the Airline Sentiment Analysis Pipeline.

> **Tip**: Training scripts take `--config`, `--model`, `--vectorizer`, `--seed` and `--binary-mode` overrides; a few serving settings can be overridden via environment variables (see [Environment Overrides](#environment-overrides)).

## Configuration Structure

```yaml
# 1. DATA LOADING
data:
  path: "data/Tweets.csv"                  # Path to CSV dataset
  target_column: "airline_sentiment"       # Column containing target labels
  text_column: "text"                      # Column containing tweet text
  confidence_columns:
    sentiment: "airline_sentiment_confidence"  # Confidence score for sentiment label
    reason: "negativereason_confidence"    # Confidence score for negative reason
  confidence_threshold: 0.7                # Filter threshold: [0.0, 1.0]

# 2. TEXT PREPROCESSING
preprocessing:
  vectorizer:
    type: "tfidf"                          # "tfidf" | "count" | "sentence_embedding" (CLI: --vectorizer)
    max_features: 2000                     # Maximum vocabulary size (tfidf/count only)
    ngram_range: [1, 2]                    # N-gram range: [min, max] (tfidf/count only)
    lowercase: true                        # Convert to lowercase before vectorizing
    stop_words: null                       # Not read by the code: stopword removal is controlled by nlp.remove_stopwords

  # cleaning / nlp apply to tfidf/count. They are embedded in the fitted vectorizer,
  # so training and inference (API, CLI, dashboard) clean text identically.
  cleaning:
    lowercase: true                        # Mirrors vectorizer.lowercase for flexibility
    remove_urls: true                      # Strip http/https/www links
    remove_mentions: true                  # Strip @username (true = remove @airline and @user mentions)
    remove_special_chars: true             # Strip everything except [a-z0-9\s!?.]
    remove_extra_whitespace: true          # Collapse multiple spaces/tabs/newlines
  
  nlp:
    lemmatize: false                       # Enable NLTK lemmatization
    remove_stopwords: false                # Remove English stopwords (vectorizer stop list; also in lemmatization)

  # Used only when vectorizer.type == "sentence_embedding"
  # Requires torch + sentence-transformers (requirements-dl.txt)
  embedding:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"  # Any Sentence-Transformers model (384-d here)
    batch_size: 64                         # Encoding batch size
    normalize: true                        # L2-normalize embeddings (recommended for linear models)
    device: "cpu"                          # "cpu" or "cuda"
    cleaning:                              # Light cleaning: casing, punctuation and emoji carry sentiment
      lowercase: false
      remove_urls: true
      remove_mentions: true                # Parity with TF-IDF: airline handles are not sentiment
      remove_special_chars: false
      remove_extra_whitespace: true

# 3. MODEL CONFIGURATION
model:
  type: "logistic_regression"              # "logistic_regression" | "mlp" (CLI: --model)
  training:                                # Shared by both model types (except max_iter)
    random_state: 42                       # Seed for split, model init and shuffling
    max_iter: 500                          # Maximum optimizer iterations (logistic regression)
    class_weight: "balanced"               # "balanced", "none", or dict
    use_confidence_weights: true           # Use annotation confidence as sample_weight
  
  regularization:                          # Logistic regression only
    solver: "lbfgs"                        # "lbfgs", "liblinear", "saga"
    penalty: "l2"                          # "l2", "l1", "elasticnet", "none"
    C: 1.0                                 # Inverse regularization strength (lower = stronger)
    l1_ratio: 0.5                          # Only for penalty="elasticnet": [0.0, 1.0]

  # PyTorch MLP — used only when type == "mlp" (requires torch)
  mlp:
    hidden_dims: [128]                     # Hidden layer sizes, e.g. [256, 64]
    dropout: 0.3                           # Dropout after each hidden layer
    learning_rate: 0.001                   # AdamW learning rate
    weight_decay: 0.0001                   # AdamW decoupled weight decay
    batch_size: 64                         # Mini-batch size (sparse TF-IDF is densified per batch only)
    max_epochs: 50                         # Upper bound on epochs
    patience: 5                            # Early stopping: epochs without validation improvement
    validation_fraction: 0.1               # Carved from the training split; the test set is never used
    device: "cpu"                          # "cpu" or "cuda"

  artifacts:
    path: "artifacts/model_bundle.joblib"  # Path to save the trained model
    auto_save: true                        # Auto-save after training completes

# 4. EVALUATION & METRICS
evaluation:
  split:
    test_size: 0.25                        # Test set fraction: (0.0, 1.0)
    stratify: true                         # Preserve class distribution in splits
  
  metrics:
    primary: ["accuracy", "f1_macro", "f1_weighted", "roc_auc_ovo"]  # Metrics to compute
  
  reporting:
    include_confusion_matrix: true         # Include confusion matrix in reports
    include_classification_report: true    # Include per-class precision/recall/f1
    export_formats: ["json", "csv"]        # Export formats: "json", "csv"
    
    plot_settings:
      normalize: true                      # Normalize confusion matrix (rates vs counts)
      cmap: "Blues"                        # Matplotlib colormap
      figsize: [8, 6]                      # Figure size in inches
    
    misclassified_examples:
      n_top: 10                            # Number of top misclassified examples
      include_text: true                   # Include original text in the report

# 5. INTERPRETABILITY
interpretability:
  weight_based:
    top_words_threshold: 2.0               # |weight| threshold for displaying top words
    n_top_features: 20                     # Number of top features to show per class
  
  shap:
    enabled: false                         # Enable SHAP explanations (requires `shap` package)
    background_samples: 100                # Number of samples for SHAP background
    random_state: 42                       # Seed for reproducible plots
  
  plot_settings:
    horizontal: true                       # Use horizontal bar charts for feature importance
    figsize: [10, 8]                       # Figure size in inches

# 6. SERVING / DASHBOARD
serving:
  api:
    enabled: true                          # Enable FastAPI server
    host: "0.0.0.0"                        # Bind address for uvicorn
    port: 8000                             # Port for uvicorn
    endpoint: "/predict"                   # Base path for prediction endpoints
    log_level: "INFO"                      # Logging level: "DEBUG", "INFO", "WARNING"
    rate_limit_per_minute: 60              # Request rate limit (optional)
    cors_origins: ["*"]                    # Allowed CORS origins (["*"] = all)
    limits:
      max_text_length: 1000                # Max characters per input text
      max_batch_size: 100                  # Max number of texts in a batch request
      min_explain_count: 1                 # Min top words in explanation
      max_explain_count: 20                # Max top words in explanation
      max_request_size_mb: 10              # Max request body size in MB
  
  dashboard:
    enabled: false                         # Enable Streamlit dashboard
    title: "Airline Sentiment Predictor"   # Dashboard window title
    framework: "streamlit"                 # "streamlit" or "gradio"
```

## Validation & Constraints

### Solver/Penalty Compatibility (scikit-learn)

| Solver | Supported Penalties | Notes |
|--------|---------------------|-------|
| `lbfgs` | `l2`, `none` | Default, recommended for multiclass |
| `liblinear` | `l1`, `l2` | Good for small datasets |
| `saga` | `l1`, `l2`, `elasticnet`, `none` | Only solver supporting `elasticnet` |
| `newton-cg`, `sag` | `l2`, `none` | Less commonly used |

> **Error**: `solver: lbfgs + penalty: l1` → `ValueError: Invalid combination`

### Validation Behavior

Configuration is checked where it is used, not up front:

- **Data loading** (`src/data_loader.py`): required `data.*` keys and the type of `data.confidence_columns`; missing dataset file → `FileNotFoundError`.
- **Vectorizer** (`create_vectorizer`): `type`, `max_features`, `ngram_range` required for `tfidf`/`count`; unknown `type` → `ValueError`.
- **Model** (`create_model`): `model.type` required; for logistic regression also `training.max_iter` and solver/penalty/`l1_ratio` compatibility; unknown `type` → `ValueError` listing supported types.
- **Split**: an empty test set (too small `test_size`) → `ValueError`.
- **Optional dependencies**: `sentence_embedding` / `mlp` without torch or sentence-transformers → `ImportError` with install instructions.

> Example: `ValueError: Invalid combination: solver=lbfgs + penalty=l1`

### Value Ranges

Valid ranges for reference. They are not checked up front: out-of-range values fail inside scikit-learn or at the split.

| Parameter | Range | Default |
|-----------|-------|---------|
| `confidence_threshold` | `[0.0, 1.0]` | `0.7` |
| `test_size` | `(0.0, 1.0)` | `0.25` |
| `C` | `(0.0, ∞)` | `1.0` |
| `l1_ratio` | `[0.0, 1.0]` | `0.5` |
| `max_text_length` | `[1, 10000]` | `1000` |
| `max_batch_size` | `[1, 1000]` | `100` |

## Environment Overrides

Only these serving settings read environment variables (`src/api/config.py`); they take precedence over `config.yaml` and apply to the API and the dashboard:

| Variable | Overrides | Example |
|----------|-----------|---------|
| `MODEL_PATH` | `model.artifacts.path` — the bundle the API/dashboard serves | `artifacts/embedding_lr/model_bundle.joblib` |
| `CORS_ORIGINS` | `serving.api.cors_origins` (JSON array) | `'["https://myapp.com"]'` |
| `API_LOG_LEVEL` | `serving.api.log_level` | `WARNING` |

```bash
MODEL_PATH=artifacts/embedding_lr/model_bundle.joblib uvicorn src.api.main:app --port 8000
```

Host and port of the running server are set by `uvicorn --host/--port` (or `streamlit run --server.port`), not by `serving.api.host/port`.
Training does not read environment variables: use a separate config file (`--config`) or the CLI flags `--model`, `--vectorizer`, `--seed`, `--binary-mode`, `--output-dir`.

## Configuration Examples

### Quick Prototype (Low Resources)
```yaml
model:
  training: { max_iter: 100, class_weight: "balanced" }
  regularization: { solver: "liblinear", penalty: "l2", C: 0.1 }
preprocessing:
  vectorizer: { max_features: 500, ngram_range: [1, 1] }
evaluation:
  split: { test_size: 0.1 }
```

### Production Setup (Maximum Quality)
```yaml
model:
  training: { max_iter: 1000, class_weight: "balanced", use_confidence_weights: true }
  regularization: { solver: "saga", penalty: "elasticnet", C: 0.5, l1_ratio: 0.3 }
preprocessing:
  vectorizer: { max_features: 5000, ngram_range: [1, 3], stop_words: "english" }
  nlp: { lemmatize: true, remove_stopwords: true }
interpretability:
  shap: { enabled: true, background_samples: 200 }
```

### Sentence Embeddings (Best Accuracy)
```yaml
preprocessing:
  vectorizer: { type: "sentence_embedding" }
  embedding: { model_name: "sentence-transformers/all-MiniLM-L6-v2", normalize: true }
model:
  type: "logistic_regression"   # the MLP gives no gain on top of embeddings (see README)
```
Equivalent without editing the config: `python scripts/train.py --vectorizer sentence_embedding`.
Word-level explanations and feature-importance plots are not available for this setup.

### PyTorch MLP
```yaml
model:
  type: "mlp"
  mlp: { hidden_dims: [256, 64], dropout: 0.3, patience: 5 }
```
Or `python scripts/train.py --model mlp` (works with any `vectorizer.type`).

### Debugging & Interpretation
```yaml
interpretability:
  weight_based: { top_words_threshold: 1.0, n_top_features: 50 }
  shap: { enabled: true }
evaluation:
  reporting:
    include_confusion_matrix: true
    misclassified_examples: { n_top: 50, include_text: true }
```

## FAQ

**Q: Why is `remove_mentions: true` by default?**  
A: Removing mentions (e.g., `@VirginAmerica`) helps reduce noise and prevents the model from overfitting to specific airline names rather than learning sentiment-bearing words. 

> **Tip**: If you need to analyze brand-specific feedback or entity-level sentiment, set `remove_mentions: false` in your config to preserve these tokens.

**Q: When should I use `elasticnet`?**  
A: When you want to combine L1 (feature selection) and L2 (weight stability) regularization. Requires `solver: saga` and tuning `l1_ratio`.

**Q: How do I disable console logging?**  
A: Set `serving.api.log_level: "WARNING"` or run with `PYTHON_LOG_LEVEL=ERROR`.

**Q: TF-IDF or sentence embeddings?**  
A: Embeddings are ~2 pp macro-F1 better on this dataset but ~40× slower per request, need torch and a 90 MB model, and give no word-level explanations. TF-IDF + logistic regression stays the default. Measured comparison: `python scripts/compare.py` and the README results table.

**Q: Tests pass but model performance is poor?**  
A: Check: 1) `confidence_threshold` isn't too high (losing data), 2) `max_features` isn't too low (losing signal), 3) `class_weight: "balanced"` is used for imbalanced datasets.