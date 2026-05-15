import json
import joblib
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from .constants import TARGET_MAPPING, TARGET_MAPPING_INV
from .utils.logging_config import setup_logger

logger = setup_logger(__name__)


def _validate_solver_penalty(
        solver: str, penalty: str, l1_ratio: Optional[float] = None
) -> None:
    """Validate that solver and penalty combination is supported by sklearn."""
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

    if penalty in ["l1", "l2", "none"] and solver in valid_combinations:
        logger.debug(
            f"Note: penalty='{penalty}' is deprecated in sklearn 1.8+. "
            "Consider migrating config to use l1_ratio directly."
        )


def create_model(config: dict) -> LogisticRegression:
    """Initialize LogisticRegression with parameters from config."""
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

    # Validate solver/penalty combination
    solver = reg_cfg.get("solver", "lbfgs")
    penalty = reg_cfg.get("penalty", "l2")
    l1_ratio_config = reg_cfg.get("l1_ratio", 0.5)

    _validate_solver_penalty(solver, penalty, l1_ratio_config)

    if penalty == "none":
        l1_ratio = None  # No regularization
        C_val = np.inf   # C=np.inf means no regularization
    elif penalty == "l1":
        l1_ratio = 1.0   # 1.0 = pure L1
        C_val = reg_cfg.get("C", 1.0)
    elif penalty == "l2":
        l1_ratio = 0.0   # 0.0 = pure L2 (explicit to avoid warning)
        C_val = reg_cfg.get("C", 1.0)
    elif penalty == "elasticnet":
        l1_ratio = l1_ratio_config  # Use config value [0.0, 1.0]
        C_val = reg_cfg.get("C", 1.0)
    else:
        # Fallback: default to L2
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
    X_train: Union[np.ndarray, spmatrix],
    y_train: np.ndarray,
    config: dict,
    sample_weights: Optional[np.ndarray] = None,
) -> LogisticRegression:
    """
    Train LogisticRegression model
    with optional confidence-based sample weights.
    """
    model_cfg = config.get("model", {})
    training_cfg = model_cfg.get("training", {})

    # Input validation
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train cannot be None")
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

    # Confidence-aware training: prioritize high-confidence labels
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
    X,
    y_true: np.ndarray,
    config: dict,
    labels: Optional[list] = None,
) -> dict:
    """Evaluate model on given data using metrics from config."""
    eval_cfg = config.get("evaluation", {})
    metrics_cfg = eval_cfg.get("metrics", {})
    reporting_cfg = eval_cfg.get("reporting", {})

    # Input validation
    if X is None or y_true is None:
        raise ValueError("X and y_true cannot be None")

    # Handle sparse matrices: use shape[0] for row count
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

    # Always compute accuracy
    results["accuracy"] = accuracy_score(y_true, y_pred)
    logger.info(f"Accuracy: {results['accuracy']:.4f}")

    # F1-score: support multiclass via average parameter
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

    # ROC-AUC: multiclass via 'ovo' (one-vs-one) or 'ovr' (one-vs-rest)
    # Note: roc_auc_score handles binary classification automatically
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

    # Confusion matrix (optional, for logging/visualization)
    if reporting_cfg.get("include_confusion_matrix", True):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        results["confusion_matrix"] = cm
        logger.debug(f"Confusion matrix:\n{cm}")

    return results


def prepare_sample_weights(
    df: pd.DataFrame,
    confidence_column: str = "sentiment_confidence",
    normalize: bool = False,  # Default False: confidence already in [0,1]
) -> np.ndarray:
    """Extract and optionally normalize confidence scores for sample_weight."""

    # Validate column exists
    if confidence_column not in df.columns:
        raise ValueError(
            f"Column '{confidence_column}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    weights = df[confidence_column].copy()

    # Handle missing values: default to neutral confidence (0.5)
    if weights.isna().any():
        na_count = weights.isna().sum()
        logger.warning(
            f"{na_count} missing values in '{confidence_column}'. "
            "Filling with default confidence=0.5"
        )
        weights = weights.fillna(0.5)

    weights = weights.values.astype(float)

    # Optional normalization (only if values outside [0,1])
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
    vectorizer,
    output_dir: Union[str, Path],
    filename: str = "model_bundle.joblib",
) -> Path:
    """Save trained model and vectorizer to disk."""
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
    model_path: Union[str, Path],
) -> Tuple[
    LogisticRegression,
    Union[CountVectorizer, TfidfVectorizer],
    dict,
    dict
]:
    """Load trained model bundle from disk."""
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
    vectorizer,
    texts: Union[str, list[str]],
    return_proba: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Predict sentiment for one or more text samples."""
    # Handle single string input
    if isinstance(texts, str):
        texts = [texts]

    # Handle empty input
    if not texts:
        logger.warning("predict_sentiment called with empty text list")
        if return_proba:
            n_classes = (
                model.classes_.shape[0] if hasattr(model, "classes_") else 3
            )
            return np.array([]), np.array([]).reshape(0, n_classes)
        return np.array([])

    # Vectorize input texts
    X = vectorizer.transform(texts)

    # Predict
    y_pred = model.predict(X)

    if return_proba:
        y_proba = model.predict_proba(X)
        return y_pred, y_proba

    return y_pred


def decode_predictions(
    predictions: np.ndarray,
    mapping_inv: Optional[dict] = None,
    strict: bool = False,  # If True, raise on unknown labels
) -> list[str]:
    """Convert encoded predictions back to sentiment labels."""
    if mapping_inv is None:
        mapping_inv = TARGET_MAPPING_INV

    result = []
    for p in predictions:
        label = mapping_inv.get(int(p))
        if label is None:
            if strict:
                raise ValueError(f"Unknown prediction label: {p}")
            logger.warning(f"Unknown prediction label {p}, using 'unknown'")
            label = "unknown"
        result.append(label)

    return result


def save_evaluation_results(
    results: dict,
    output_dir: Union[str, Path],
    filename: str = "evaluation_metrics.json",
) -> Path:
    """Save evaluation metrics to JSON file."""
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable values (e.g., numpy arrays)
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
