"""
Core ML module for model lifecycle management.

Handles LogisticRegression creation, training with confidence-aware weights,
evaluation with multi-metric support, persistence via joblib, and prediction.
All functions validate inputs explicitly and log actionable errors.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .constants import TARGET_MAPPING, TARGET_MAPPING_INV
from .utils.logging_config import setup_logger

logger = setup_logger(__name__)


def _validate_solver_penalty(
        solver: str, penalty: str, l1_ratio: float | None = None
) -> None:
    """
    Validate solver/penalty combination against scikit-learn constraints.

    Args:
        solver: Optimization algorithm (lbfgs, liblinear, saga, etc.)
        penalty: Regularization type (l1, l2, elasticnet, none)
        l1_ratio: Mixing parameter for elasticnet [0.0, 1.0]

    Raises:
        ValueError: If combination is unsupported or l1_ratio out of range

    Note:
        scikit-learn restricts solver/penalty pairs:
        - lbfgs/newton-cg/sag: only l2 or none
        - liblinear: l1 or l2 (good for small datasets)
        - saga: all penalties including elasticnet (only solver supporting l1)
    """
    # Source of truth for valid combinations
    valid_combinations = {
        "lbfgs": {"l2", "none"},
        "liblinear": {"l1", "l2"},
        "saga": {"l1", "l2", "elasticnet", "none"},
        "newton-cg": {"l2", "none"},
        "sag": {"l2", "none"},
    }

    if solver not in valid_combinations:
        raise ValueError(f"Unsupported solver: {solver}")

    if penalty not in valid_combinations[solver]:
        raise ValueError(
            f"Invalid combination: solver='{solver}' "
            f"does not support penalty='{penalty}'. "
            f"Valid penalties for '{solver}': "
            f"{sorted(valid_combinations[solver])}"
        )

    if penalty == "elasticnet":
        if solver != "saga":
            raise ValueError("penalty='elasticnet' requires solver='saga'")
        if l1_ratio is not None and not (0.0 <= l1_ratio <= 1.0):
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")

    # Deprecation warning for future sklearn compatibility
    if penalty in ["l1", "l2", "none"] and solver in valid_combinations:
        logger.debug(
            f"Note: penalty='{penalty}' is deprecated in sklearn 1.8+. "
            "Consider migrating config to use l1_ratio directly."
        )


def create_model(config: dict) -> LogisticRegression:
    """
    Initialize LogisticRegression with parameters from configuration.

    Args:
        config: Dict with model/training/regularization settings

    Returns:
        Configured LogisticRegression instance (unfitted)

    Raises:
        ValueError: If required config keys are missing
        or model type unsupported

    Note:
        Penalty → (l1_ratio, C) mapping:
        - none: l1_ratio=None, C=np.inf (no regularization)
        - l1: l1_ratio=1.0, C=config value
        - l2: l1_ratio=0.0, C=config value (explicit to avoid sklearn warning)
        - elasticnet: l1_ratio=config value, C=config value
    """
    model_cfg = config.get("model", {})
    training_cfg = model_cfg.get("training", {})
    reg_cfg = model_cfg.get("regularization", {})

    if "type" not in model_cfg:
        raise ValueError("Missing required config key: model.type")
    if "max_iter" not in training_cfg:
        raise ValueError(
            "Missing required config key: model.training.max_iter"
        )

    if model_cfg["type"] != "logistic_regression":
        raise ValueError(
            f"Unsupported model_type: {model_cfg['type']}. "
            "Only 'logistic_regression' is supported in this version."
        )

    # Handle class_weight: "balanced", "none" → None, or dict
    class_weight = training_cfg.get("class_weight", "balanced")
    if class_weight == "none":
        class_weight = None

    logger.info(
        f"Initializing LogisticRegression with: "
        f"max_iter={training_cfg['max_iter']}, class_weight={class_weight}"
    )

    solver = reg_cfg.get("solver", "lbfgs")
    penalty = reg_cfg.get("penalty", "l2")
    l1_ratio_config = reg_cfg.get("l1_ratio", 0.5)

    _validate_solver_penalty(solver, penalty, l1_ratio_config)

    # Map penalty enum to sklearn parameters
    if penalty == "none":
        l1_ratio = None  # No regularization
        C_val = np.inf   # C=np.inf disables regularization in sklearn
    elif penalty == "l1":
        l1_ratio = 1.0   # Pure L1
        C_val = reg_cfg.get("C", 1.0)
    elif penalty == "l2":
        l1_ratio = 0.0   # Pure L2 (explicit to avoid warning)
        C_val = reg_cfg.get("C", 1.0)
    elif penalty == "elasticnet":
        l1_ratio = l1_ratio_config  # Use config value [0.0, 1.0]
        C_val = reg_cfg.get("C", 1.0)
    else:
        # Fallback: default to L2 regularization
        l1_ratio = 0.0
        C_val = reg_cfg.get("C", 1.0)

    logger.debug(
        f"Regularization config: solver={solver}, penalty={penalty}, "
        f"C={C_val}, l1_ratio={l1_ratio}"
    )

    model_kwargs = {
        "max_iter": training_cfg["max_iter"],
        "class_weight": class_weight,
        "random_state": training_cfg.get("random_state", 42),
        "solver": solver,
        "C": C_val,
    }

    if l1_ratio is not None:
        model_kwargs["l1_ratio"] = l1_ratio

    logger.debug(
        f"Model kwargs: {model_kwargs}",
        extra={"model_init_params": model_kwargs}
    )

    logger.info(
        f"LogisticRegression initialized: solver={solver}, "
        f"penalty={penalty}, C={C_val}, l1_ratio={l1_ratio}, "
        f"max_iter={training_cfg['max_iter']}"
    )

    return LogisticRegression(**model_kwargs)


def train_model(
    X_train: np.ndarray | spmatrix,
    y_train: np.ndarray,
    config: dict,
    sample_weights: np.ndarray | None = None,
) -> LogisticRegression:
    """
    Train LogisticRegression with optional confidence-based sample weighting.

    Args:
        X_train: Feature matrix (dense numpy array or scipy sparse)
        y_train: Target labels (1D numpy array)
        config: Configuration dict with training settings
        sample_weights: Optional per-sample weights
        (e.g., annotation confidence)

    Returns:
        Fitted LogisticRegression model

    Raises:
        ValueError: If inputs are None or shape mismatch detected

    Note:
        Supports both dense (np.ndarray) and sparse (scipy) matrices.
        Confidence-aware training:
        higher-confidence annotations have more influence.
    """
    model_cfg = config.get("model", {})
    training_cfg = model_cfg.get("training", {})

    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train cannot be None")
    
    # Handle both dense and sparse matrix shapes
    n_samples_X = (
        X_train.shape[0] if hasattr(X_train, "shape") else len(X_train)
    )
    n_samples_y = len(y_train)

    if n_samples_X != n_samples_y:
        raise ValueError(
            f"Shape mismatch: X_train has {n_samples_X} samples, "
            f"y_train has {n_samples_y} labels"
        )

    if sample_weights is not None:
        n_samples_w = (
            sample_weights.shape[0]
            if hasattr(sample_weights, "shape")
            else len(sample_weights)
        )
        if n_samples_w != n_samples_y:
            raise ValueError(
                f"sample_weights length ({n_samples_w}) != "
                f"y_train length ({n_samples_y})"
            )

    model = create_model(config)

    # Confidence-aware training prioritizes high-quality annotations
    if training_cfg.get(
        "use_confidence_weights", True
    ) and sample_weights is not None:
        logger.info(
            "Training with confidence-based sample weights "
            f"(n={len(sample_weights)})"
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        logger.info("Training without sample weights")
        model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: LogisticRegression,
    X: np.ndarray | spmatrix,
    y_true: np.ndarray,
    config: dict,
    labels: list | None = None,
) -> dict:
    """
    Evaluate model using metrics specified in configuration.

    Args:
        model: Fitted LogisticRegression instance
        X: Feature matrix for evaluation
        y_true: Ground truth labels
        config: Configuration dict with metrics/reporting settings
        labels: Optional list of class labels for confusion matrix ordering

    Returns:
        Dict with computed metrics (accuracy, f1_macro, roc_auc_ovo, etc.)

    Raises:
        ValueError: If inputs are None or shape mismatch

    Note:
        - ROC-AUC uses one-vs-one (ovo) or one-vs-rest (ovr) for multiclass
        - zero_division=0 prevents warnings on empty class slices
        - Empty dataset returns {} with warning (graceful degradation)
    """
    eval_cfg = config.get("evaluation", {})
    metrics_cfg = eval_cfg.get("metrics", {})
    reporting_cfg = eval_cfg.get("reporting", {})

    if X is None or y_true is None:
        raise ValueError("X and y_true cannot be None")

    # Handle sparse matrix row count
    n_samples_X = X.shape[0] if hasattr(X, "shape") else len(X)
    n_samples_y = len(y_true)

    if n_samples_X != n_samples_y:
        raise ValueError(
            f"Shape mismatch: X has {n_samples_X} samples, "
            f"y_true has {n_samples_y} labels"
        )

    if len(y_true) == 0:
        logger.warning("evaluate_model called with empty dataset")
        return {}

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    results = {}

    # Accuracy is always computed as baseline metric
    results["accuracy"] = accuracy_score(y_true, y_pred)
    logger.info(f"Accuracy: {results['accuracy']:.4f}")

    # F1 with macro/weighted averaging for multiclass support
    metrics = metrics_cfg.get("primary", ["accuracy"])
    if "f1_macro" in metrics:
        results["f1_macro"] = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        logger.info(f"F1 (macro): {results['f1_macro']:.4f}")

    if "f1_weighted" in metrics:
        results["f1_weighted"] = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        logger.info(f"F1 (weighted): {results['f1_weighted']:.4f}")

    # ROC-AUC with graceful fallback on empty proba or computation errors
    if y_proba.size == 0:
        logger.warning("Skipping ROC-AUC: empty probability array")
    else:
        if "roc_auc_ovo" in metrics:
            try:
                results["roc_auc_ovo"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovo", average="macro"
                )
                logger.info(f"ROC-AUC (OvO): {results['roc_auc_ovo']:.4f}")
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC (OvO): {e}")

        if "roc_auc_ovr" in metrics:
            try:
                results["roc_auc_ovr"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
                logger.info(f"ROC-AUC (OvR): {results['roc_auc_ovr']:.4f}")
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC (OvR): {e}")

    if reporting_cfg.get("include_confusion_matrix", True):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        results["confusion_matrix"] = cm
        logger.debug(f"Confusion matrix:\n{cm}")

    return results


def prepare_sample_weights(
    df: pd.DataFrame,
    confidence_column: str = "sentiment_confidence",
    normalize: bool = False,
) -> np.ndarray:
    """
    Extract and optionally normalize confidence scores for sample_weight.

    Args:
        df: DataFrame containing confidence column
        confidence_column: Name of column with confidence scores
        normalize: If True, scale values to [0, 1] (only if outside range)

    Returns:
        1D numpy array of sample weights

    Raises:
        ValueError: If confidence_column not found in DataFrame

    Note:
        - Missing values filled with 0.5 (neutral confidence)
        - Normalization uses min-max: (x - min) / (max - min)
        - Confidence scores from dataset are typically already in [0, 1]
    """
    if confidence_column not in df.columns:
        raise ValueError(
            f"Column '{confidence_column}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    weights = df[confidence_column].copy()

    if weights.isna().any():
        na_count = weights.isna().sum()
        logger.warning(
            f"{na_count} missing values in '{confidence_column}'. "
            "Filling with default confidence=0.5"
        )
        weights = weights.fillna(0.5)

    weights = weights.values.astype(float)

    # Min-max normalization only if values exceed [0, 1] range
    if normalize and (weights.min() < 0 or weights.max() > 1):
        min_val, max_val = weights.min(), weights.max()
        if max_val > min_val:
            weights = (weights - min_val) / (max_val - min_val)
        logger.debug(
            "Normalized sample weights: range "
            f"[{weights.min():.3f}, {weights.max():.3f}]"
        )

    return weights


def save_model(
    model: LogisticRegression,
    vectorizer: CountVectorizer | TfidfVectorizer,
    output_dir: str | Path,
    filename: str = "model_bundle.joblib",
) -> Path:
    """
    Save trained model, vectorizer, and mappings to disk via joblib.

    Args:
        model: Fitted LogisticRegression instance
        vectorizer: Fitted CountVectorizer or TfidfVectorizer
        output_dir: Directory to save bundle (created if missing)
        filename: Output filename (default: model_bundle.joblib)

    Returns:
        Path to saved file

    Note:
        Bundle includes: model, vectorizer, TARGET_MAPPING, TARGET_MAPPING_INV
        Compression level 3 balances size and load speed.
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "target_mapping": TARGET_MAPPING,
        "target_mapping_inv": TARGET_MAPPING_INV,
    }

    joblib.dump(bundle, output_path, compress=3)
    logger.info(f"Model bundle saved to {output_path}")
    return output_path


def load_model(
    model_path: str | Path,
) -> tuple[
    LogisticRegression,
    CountVectorizer | TfidfVectorizer,
    dict,
    dict
]:
    """
    Load trained model bundle from disk.

    Args:
        model_path: Path to .joblib file

    Returns:
        Tuple of (model, vectorizer, target_mapping, target_mapping_inv)

    Raises:
        FileNotFoundError: If path doesn't exist
        joblib.exception: If file is corrupted or incompatible
    """
    bundle = joblib.load(model_path)
    logger.info(f"Model bundle loaded from {model_path}")
    return (
        bundle["model"],
        bundle["vectorizer"],
        bundle["target_mapping"],
        bundle["target_mapping_inv"],
    )


def predict_sentiment(
    model: LogisticRegression,
    vectorizer: CountVectorizer | TfidfVectorizer,
    texts: str | list[str],
    return_proba: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Predict sentiment for one or more text samples.

    Args:
        model: Fitted LogisticRegression instance
        vectorizer: Fitted vectorizer for text transformation
        texts: Single string or list of strings to classify
        return_proba: If True, return (predictions, probabilities);
        else predictions only

    Returns:
        np.ndarray of predicted class indices, or tuple with probabilities

    Note:
        - Handles empty input gracefully (returns empty arrays)
        - Supports sparse matrix output from vectorizer.transform()
        - Probabilities shape: (n_samples, n_classes)
    """
    # Normalize single string to list for uniform processing
    if isinstance(texts, str):
        texts = [texts]

    # Handle empty input with shaped empty arrays
    if not texts:
        logger.warning("predict_sentiment called with empty text list")
        if return_proba:
            n_classes = (
                model.classes_.shape[0] if hasattr(model, "classes_") else 3
            )
            return np.array([]), np.array([]).reshape(0, n_classes)
        return np.array([])

    X = vectorizer.transform(texts)
    y_pred = model.predict(X)

    if return_proba:
        y_proba = model.predict_proba(X)
        return y_pred, y_proba

    return y_pred


def decode_predictions(
    predictions: np.ndarray,
    mapping_inv: dict | None = None,
    strict: bool = False,
) -> list[str]:
    """
    Convert encoded integer predictions back to human-readable labels.

    Args:
        predictions: Array of integer class indices
        mapping_inv: Optional custom index→label mapping
        (default: TARGET_MAPPING_INV)
        strict: If True, raise ValueError on unknown labels;
        else log warning and use 'unknown'

    Returns:
        List of string labels corresponding to input predictions

    Raises:
        ValueError: If strict=True and unknown label encountered

    Note:
        - strict=False: unknown labels → 'unknown' + warning (production-safe)
        - strict=True: unknown labels → raise ValueError (debug/testing)
    """
    if mapping_inv is None:
        mapping_inv = TARGET_MAPPING_INV

    result = []
    for p in predictions:
        label = mapping_inv.get(int(p))
        if label is None:
            if strict:
                raise ValueError(f"Unknown prediction label: {p}")
            # Graceful fallback for unseen classes in production
            logger.warning(f"Unknown prediction label {p}, using 'unknown'")
            label = "unknown"
        result.append(label)

    return result


def save_evaluation_results(
    results: dict,
    output_dir: str | Path,
    filename: str = "evaluation_metrics.json",
) -> Path:
    """
    Save evaluation metrics to JSON file with numpy type conversion.

    Args:
        results: Dict of metric names to values (may include numpy types)
        output_dir: Directory to save file (created if missing)
        filename: Output filename (default: evaluation_metrics.json)

    Returns:
        Path to saved JSON file

    Note:
        Converts numpy arrays → lists, numpy scalars → float
        for JSON compatibility.
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Recursive conversion of numpy types to JSON-serializable Python types
    serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        else:
            serializable[k] = v

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"Evaluation metrics saved to {output_path}")
    return output_path
