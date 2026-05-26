# Configuration Guide

Comprehensive guide to configuring `configs/config.yaml` for the Airline Sentiment Analysis Pipeline.

> **Tip**: All values can be overridden via environment variables (see [Environment Overrides](#-environment-overrides)).

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
    type: "tfidf"                          # "tfidf" or "count"
    max_features: 2000                     # Maximum vocabulary size
    ngram_range: [1, 2]                    # N-gram range: [min, max]
    lowercase: true                        # Convert to lowercase before vectorizing
    stop_words: null                       # "english" or null (keep stopwords)
  
  cleaning:
    lowercase: true                        # Mirrors vectorizer.lowercase for flexibility
    remove_urls: true                      # Strip http/https/www links
    remove_mentions: false                 # Strip @username (false = keep @airline)
    remove_special_chars: true             # Strip everything except [a-z0-9\s!?.]
    remove_extra_whitespace: true          # Collapse multiple spaces/tabs/newlines
  
  nlp:
    lemmatize: false                       # Enable NLTK lemmatization
    remove_stopwords: false                # Remove stopwords (only if lemmatize=true)

# 3. MODEL CONFIGURATION
model:
  type: "logistic_regression"              # Currently the only supported algorithm
  training:
    random_state: 42                       # Seed for reproducibility
    max_iter: 500                          # Maximum optimizer iterations
    class_weight: "balanced"               # "balanced", "none", or dict
    use_confidence_weights: true           # Use annotation confidence as sample_weight
  
  regularization:
    solver: "lbfgs"                        # "lbfgs", "liblinear", "saga"
    penalty: "l2"                          # "l2", "l1", "elasticnet", "none"
    C: 1.0                                 # Inverse regularization strength (lower = stronger)
    l1_ratio: 0.5                          # Only for penalty="elasticnet": [0.0, 1.0]
  
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

Configuration is validated at two stages:

1. **At startup** (FastAPI/Streamlit):  
   - Type checks (int vs str)  
   - Range checks (e.g., `confidence_threshold ∈ [0.0, 1.0]`)  
   - Required keys presence  

2. **At runtime** (training/prediction):  
   - Solver/penalty compatibility (scikit-learn constraints)  
   - File path existence (`model.artifacts.path`)  
   - Resource availability (SHAP package for `interpretability.shap.enabled: true`)

> Invalid configurations raise `ValueError` with a descriptive message.  
> Example: `ValueError: Invalid combination: solver=lbfgs + penalty=l1`

### Value Ranges

| Parameter | Range | Default |
|-----------|-------|---------|
| `confidence_threshold` | `[0.0, 1.0]` | `0.7` |
| `test_size` | `(0.0, 1.0)` | `0.25` |
| `C` | `(0.0, ∞)` | `1.0` |
| `l1_ratio` | `[0.0, 1.0]` | `0.5` |
| `max_text_length` | `[1, 10000]` | `1000` |
| `max_batch_size` | `[1, 1000]` | `100` |

## Environment Overrides

Any configuration value can be overridden via environment variables:

```bash
# Examples:
export MODEL_PATH="/mnt/models/v2/model_bundle.joblib"
export API_PORT=8080
export PREPROCESSING_VECTORIZER_MAX_FEATURES=5000
export EVALUATION_SPLIT_TEST_SIZE=0.2

# Run with overrides:
python scripts/train.py  # Uses env vars on top of config.yaml
```

> **Naming Convention**:  
> `SECTION_SUBSECTION_..._KEY` in uppercase, separated by underscores.  
> Example: `preprocessing.vectorizer.max_features` → `PREPROCESSING_VECTORIZER_MAX_FEATURES`

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

**Q: Why is `remove_mentions: false` by default?**  
A: Airline mentions (e.g., `@VirginAmerica`) often carry semantic weight for sentiment. Remove them only if analyzing general sentiment rather than brand-specific feedback.

**Q: When should I use `elasticnet`?**  
A: When you want to combine L1 (feature selection) and L2 (weight stability) regularization. Requires `solver: saga` and tuning `l1_ratio`.

**Q: How do I disable console logging?**  
A: Set `serving.api.log_level: "WARNING"` or run with `PYTHON_LOG_LEVEL=ERROR`.

**Q: Tests pass but model performance is poor?**  
A: Check: 1) `confidence_threshold` isn't too high (losing data), 2) `max_features` isn't too low (losing signal), 3) `class_weight: "balanced"` is used for imbalanced datasets.