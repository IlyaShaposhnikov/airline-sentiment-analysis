"""
Model interpretability module for explaining predictions.

Provides weight-based feature attribution for LogisticRegression,
optional SHAP integration for advanced explanations, and visualization
utilities. All functions gracefully degrade if SHAP is unavailable.
"""
import importlib
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Optional dependency, imported LAZILY: shap pulls in numba/llvmlite (slow
# import, heavy native libraries). Only SHAP explanations need it, so regular
# training runs, the CLI and the API never load it.
# SHAP_AVAILABLE only checks that the package is installed (no import).


def _shap_installed() -> bool:
    try:
        return importlib.util.find_spec("shap") is not None
    except (ImportError, ValueError):
        # ValueError: sys.modules["shap"] is None (import explicitly blocked)
        return False


SHAP_AVAILABLE = _shap_installed()
_shap_module = None


def _get_shap():
    """
    Import shap on first use.

    Returns:
        The shap module, or None if it cannot be imported (e.g. a broken
        native dependency) — callers then fall back to weight explanations.
    """
    global _shap_module
    if _shap_module is None:
        try:
            _shap_module = importlib.import_module("shap")
        except Exception as e:  # ImportError, OSError from llvmlite, ...
            logger.warning(f"SHAP import failed, using weights instead: {e}")
            return None
    return _shap_module


def supports_word_explanations(vectorizer, model=None) -> bool:
    """
    Whether word-level explanations are meaningful for this model bundle.

    Weight/SHAP explanations map linear model coefficients (``coef_``) back
    to vocabulary terms via ``get_feature_names_out()``. They are not
    available when:
    - the vectorizer has no vocabulary (dense sentence embeddings: their
      dimensions are not words), or
    - the model is not linear in the features (e.g. an MLP declares
      ``_word_explanations_supported = False``).
    Callers must then skip word-level explanations and feature plots.

    Args:
        vectorizer: Fitted vectorizer from the model bundle
        model: Fitted classifier (optional; only the vectorizer is checked
            when omitted)

    Returns:
        True if word-level explanations can be computed
    """
    has_vocabulary = hasattr(vectorizer, "get_feature_names_out") or hasattr(
        vectorizer, "get_feature_names"
    )
    model_ok = model is None or getattr(
        model, "_word_explanations_supported", True
    )
    return bool(has_vocabulary and model_ok)


def _to_dense(values) -> np.ndarray:
    """Convert SHAP output (possibly scipy sparse) to a dense array."""
    if hasattr(values, "toarray"):
        return values.toarray()
    return np.asarray(values)


def _shap_values_for_class(
    shap_values, class_pos: int, n_classes: int
) -> np.ndarray:
    """
    Per-feature SHAP values of the FIRST sample for one class.

    Normalizes the formats ``LinearExplainer.shap_values`` returns:
    - list with one (n_samples, n_features) array per class
      (shap < 0.45, multiclass);
    - ndarray (n_samples, n_features, n_classes) (shap >= 0.45, multiclass);
    - ndarray (n_samples, n_features): a single output, i.e. binary logistic
      regression, where values push towards class 1. For class 0 they are
      negated — the same convention as the weight-based explanations.

    Args:
        shap_values: Raw output of ``explainer.shap_values(X)``
        class_pos: Column position of the class in ``model.classes_``
        n_classes: Number of classes of the model

    Returns:
        1D array of length n_features
    """
    if isinstance(shap_values, list):
        return _to_dense(shap_values[class_pos])[0].ravel()
    values = _to_dense(shap_values)
    if values.ndim == 3:
        return values[0, :, class_pos]
    row = values[0].ravel()
    if n_classes == 2 and class_pos == 0:
        return -row
    return row


def get_top_features_by_weight(
    model: LogisticRegression,
    vectorizer: CountVectorizer | TfidfVectorizer,
    n_top: int = 20,
    threshold: float | None = None,
    class_names: list[str] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """
    Extract top features (words) by model weight for each class.

    For LogisticRegression, coef_[class_idx, feature_idx] represents
    the weight of that feature for predicting that class.

    Args:
        model: Trained LogisticRegression instance
        vectorizer: Fitted vectorizer (to map feature indices to words)
        n_top: Number of top positive/negative features to return per class
        threshold: Optional absolute weight threshold to filter results
        class_names: Optional list of class names for output keys

    Returns:
        Dict mapping class name to dict with 'top_positive', 'top_negative',
        and 'n_features_above_threshold' keys.

    Note:
        Binary classification handling: sklearn returns single coef row
        representing decision boundary. Positive class (idx=1) uses +coef,
        negative class (idx=0) uses -coef for intuitive interpretation.
    """
    # Compatibility with sklearn <1.0 vs >=1.0
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions
        feature_names = np.array(vectorizer.get_feature_names())

    coef = model.coef_
    classes = model.classes_

    results = {}

    for idx, class_label in enumerate(classes):
        class_key = (
            class_names[idx]
            if class_names and idx < len(class_names)
            else str(class_label)
        )

        # Binary coef shape is [1, n_features], not [2, n_features]
        if len(classes) == 2 and coef.shape[0] == 1:
            # Positive class (idx=1) = +coef[0],
            # Negative class (idx=0) = -coef[0]
            logger.debug(
                f"Binary classification detected: inverting weights for "
                f"negative class (idx={idx})"
            )
            weights = coef[0] if idx == 1 else -coef[0]
        else:
            weights = coef[idx]

        word_weights = list(zip(feature_names, weights))

        if threshold is not None:
            word_weights = [
                (w, wt) for w, wt in word_weights if abs(wt) >= threshold
            ]

        word_weights_sorted = sorted(
            word_weights, key=lambda x: abs(x[1]), reverse=True
        )

        top_positive = [
            (w, wt) for w, wt in word_weights_sorted if wt > 0
        ][:n_top]
        top_negative = [
            (w, wt) for w, wt in word_weights_sorted if wt < 0
        ][:n_top]

        results[class_key] = {
            "top_positive": top_positive,
            "top_negative": top_negative,
            "n_features_above_threshold": (
                sum(1 for _, wt in word_weights if abs(wt) >= threshold)
                if threshold else len(word_weights)
            ),
        }

        logger.debug(
            f"Class '{class_key}': {len(top_positive)} positive, "
            f"{len(top_negative)} negative features (threshold={threshold})"
        )

    return results


def plot_feature_importance(
    model: LogisticRegression,
    vectorizer: CountVectorizer | TfidfVectorizer,
    class_idx: int = 0,
    n_top: int = 20,
    horizontal: bool = True,
    figsize: tuple[int, int] = (10, 8),
    save_path: str | Path | None = None,
    title: str | None = None,
    class_names: list[str] | None = None,
) -> plt.Figure:
    """
    Plot top features by weight for a specific class.

    Args:
        model: Trained LogisticRegression instance
        vectorizer: Fitted vectorizer
        class_idx: Index of class to visualize (0-based)
        n_top: Number of top features to show (positive + negative)
        horizontal: If True, plot horizontal bar chart (better for long words)
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save the figure (creates parent dirs)
        title: Optional plot title (auto-generated if None)
        class_names: Optional list of class names for title/labels

    Returns:
        Matplotlib Figure object

    Note:
        Features colored green (positive weight) or red (negative weight)
        to indicate direction of influence on prediction.
    """
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = np.array(vectorizer.get_feature_names())

    classes = model.classes_

    if len(classes) == 2 and model.coef_.shape[0] == 1:
        # Binary: single row represents decision boundary
        # Positive class (idx=1) = +coef[0], Negative class (idx=0) = -coef[0]
        coef = model.coef_[0] if class_idx == 1 else -model.coef_[0]
    else:
        # Multiclass: normal access
        coef = model.coef_[class_idx]

    class_label = (
        class_names[class_idx]
        if class_names and class_idx < len(class_names)
        else f"Class {classes[class_idx]}"
    )

    # Select top features by absolute weight magnitude
    top_indices = np.argsort(np.abs(coef))[-n_top:]
    top_features = feature_names[top_indices]
    top_weights = coef[top_indices]

    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        y_pos = np.arange(len(top_features))
        colors = ["green" if w > 0 else "red" for w in top_weights]
        ax.barh(y_pos, top_weights, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel("Weight")
        ax.set_title(title or f"Top {n_top} Features for '{class_label}'")
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    else:
        x_pos = np.arange(len(top_features))
        colors = ["green" if w > 0 else "red" for w in top_weights]
        ax.bar(x_pos, top_weights, color=colors, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Weight")
        ax.set_title(title or f"Top {n_top} Features for '{class_label}'")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    ax.grid(axis="x" if horizontal else "y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")

    return fig


def _get_shap_explanation(
    model: LogisticRegression,
    vectorizer: CountVectorizer | TfidfVectorizer,
    text: str,
    background_data: np.ndarray | None = None,
    class_pos: int = 0,
) -> dict | None:
    """
    Internal: compute SHAP explanation if SHAP is available.

    Args:
        model: Trained LogisticRegression instance
        vectorizer: Fitted vectorizer
        text: Input text to explain
        background_data: Optional background data for expected value estimation
        class_pos: Position in ``model.classes_`` of the class to explain

    Returns:
        Dict with ``values`` (1D, one per vocabulary feature) and
        ``base_value`` for that class, or None if unavailable/failed

    Note:
        Uses LinearExplainer for efficient exact SHAP values on linear models.
        Falls back to zero baseline if no background data provided
        (less accurate).
    """
    shap = _get_shap() if SHAP_AVAILABLE else None
    if shap is None:
        return None

    try:
        X_input = vectorizer.transform([text])

        # Background data needed for expected value;
        # zero fallback degrades accuracy
        if background_data is None:
            logger.warning(
                "No background data for SHAP - using zero baseline. "
                "For better results, pass a sample of training data."
            )
            background_data = np.zeros((1, X_input.shape[1]))

        explainer = shap.LinearExplainer(
            model, background_data
        )
        shap_values = explainer.shap_values(X_input)
        values = _shap_values_for_class(
            shap_values, class_pos, len(model.classes_)
        )

        base_val = getattr(explainer, "expected_value", 0.0)
        base_arr = np.atleast_1d(np.asarray(base_val, dtype=float))
        if base_arr.size > 1:
            base_val = float(base_arr[class_pos])
        else:
            base_val = float(base_arr[0])
            if len(model.classes_) == 2 and class_pos == 0:
                base_val = -base_val
        return {"values": values, "base_value": base_val}

    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return None


def explain_prediction(
    model: LogisticRegression,
    vectorizer: CountVectorizer | TfidfVectorizer,
    text: str,
    class_names: list[str] | None = None,
    n_top_contributors: int = 10,
    use_shap: bool = False,
    background_data: np.ndarray | None = None,
) -> dict:
    """
    Explain a single prediction using model weights or SHAP values.

    Args:
        model: Trained LogisticRegression instance
        vectorizer: Fitted vectorizer
        text: Input text to explain
        class_names: Optional list of class names for readable output
        n_top_contributors: Number of top contributing words to return
        use_shap: If True and SHAP available, use SHAP; else use weights
        background_data: Optional background data for SHAP (improves accuracy)

    Returns:
        Dict with text, predicted_class, probabilities, top_contributors,
        and explanation method ('shap' or 'weights').

    Note:
        - SHAP: exact feature contributions via LinearExplainer (if available)
        - Weights: model.coef_ values for features present in input text
        - Binary: negative class weights inverted for intuitive interpretation
        - Graceful fallback: if SHAP fails, automatically uses weights
    """
    if not supports_word_explanations(vectorizer, model):
        logger.warning(
            "Word-level explanations are not supported for "
            f"{type(model).__name__} + {type(vectorizer).__name__}"
        )
        return {
            "error": "Explanations not supported for this model",
            "method": "none",
            "predicted_class": None,
            "probabilities": {},
            "top_contributors": [],
        }

    if not text or not text.strip():
        logger.warning("explain_prediction called with empty text")
        return {
            "error": "Empty input text",
            "method": "none",
            "predicted_class": None,
            "probabilities": {},
            "top_contributors": [],
        }

    X_input = vectorizer.transform([text])
    pred_idx = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    classes = model.classes_
    pred_class_name = (
        class_names[pred_idx]
        if class_names and pred_idx < len(class_names)
        else str(pred_idx)
    )

    result = {
        "text": text,
        "predicted_class": pred_class_name,
        "predicted_class_idx": int(pred_idx),
        "probabilities": {
            (
                class_names[i]
                if class_names and i < len(class_names)
                else str(classes[i])
            ): float(p)
            for i, p in enumerate(proba)
        },
        "top_contributors": [],
        "method": "weights",
    }

    # Try SHAP first if requested; fallback to weights on any failure
    if use_shap and SHAP_AVAILABLE:
        class_pos = int(np.flatnonzero(classes == pred_idx)[0])
        shap_result = _get_shap_explanation(
            model, vectorizer, text, background_data, class_pos=class_pos
        )
        shap_vals = (
            None if shap_result is None
            else np.asarray(shap_result.get("values")).ravel()
        )
        try:
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            feature_names = np.array(vectorizer.get_feature_names())

        if shap_vals is not None and len(shap_vals) != len(feature_names):
            # Never pair values with the wrong words: use weights instead
            logger.warning(
                f"SHAP returned {len(shap_vals)} values for "
                f"{len(feature_names)} features; using weights instead"
            )
            shap_vals = None

        if shap_vals is not None:
            result["method"] = "shap"
            nonzero_indices = np.flatnonzero(np.abs(shap_vals) > 1e-10)
            contributors = [
                (feature_names[i], float(shap_vals[i]))
                for i in nonzero_indices
            ]
            contributors_sorted = sorted(
                contributors, key=lambda x: abs(x[1]), reverse=True
            )
            result["top_contributors"] = (
                contributors_sorted[:n_top_contributors]
            )
            logger.debug(
                "SHAP explanation: "
                f"{len(result['top_contributors'])} contributors"
            )
            return result

    # Fallback to weight-based explanation (always available for linear models)
    classes = model.classes_
    if len(classes) == 2 and model.coef_.shape[0] == 1:
        # Binary: invert weights for negative class
        # for intuitive interpretation
        weights = model.coef_[0] if pred_idx == 1 else -model.coef_[0]
    else:
        weights = model.coef_[pred_idx]

    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = np.array(vectorizer.get_feature_names())

    # Only consider features actually present in the input text
    input_features = X_input.indices
    input_weights = weights[input_features]
    input_names = feature_names[input_features]

    contributors = [
        (name, float(wt)) for name, wt in zip(input_names, input_weights)
        if wt != 0
    ]

    contributors_sorted = sorted(
        contributors, key=lambda x: abs(x[1]), reverse=True
    )
    result["top_contributors"] = contributors_sorted[:n_top_contributors]

    logger.debug(
        "Weight-based explanation: "
        f"{len(result['top_contributors'])} contributors"
    )
    return result


def plot_shap_summary(
    model: LogisticRegression,
    vectorizer: CountVectorizer | TfidfVectorizer,
    texts: list[str],
    class_idx: int | None = None,
    n_top: int = 20,
    save_path: str | Path | None = None,
    random_state: int | None = None,
) -> plt.Figure | None:
    """
    Plot SHAP summary plot for multiple examples (if SHAP available).

    Args:
        model: Trained LogisticRegression instance
        vectorizer: Fitted vectorizer
        texts: List of input texts to explain
        class_idx: Optional class index to focus on (for multiclass)
        n_top: Number of top features to display in summary
        save_path: Optional path to save the figure
        random_state: Optional seed for reproducible background sampling

    Returns:
        Matplotlib Figure object or None if SHAP not available/failed

    Note:
        Uses random subset of texts as background data for efficiency.
        For multiclass, plots specified class_idx or first class by default.
    """
    shap = _get_shap() if SHAP_AVAILABLE else None
    if shap is None:
        logger.warning("SHAP not available - skipping summary plot")
        return None

    try:
        X = vectorizer.transform(texts)

        # Use random subset for background to balance accuracy/speed
        rng = np.random.default_rng(random_state)
        background_idx = rng.choice(
            len(texts), min(100, len(texts)), replace=False
        )
        background = X[background_idx]

        explainer = shap.LinearExplainer(
            model, background
        )
        shap_values = explainer.shap_values(X)

        # Multiclass: list per class (shap < 0.45) or a trailing class
        # axis (shap >= 0.45); select the target class
        target = class_idx or 0
        if isinstance(shap_values, list):
            target = target if target < len(shap_values) else 0
            values_to_plot = _to_dense(shap_values[target])
        else:
            values_to_plot = _to_dense(shap_values)
            if values_to_plot.ndim == 3:
                if target >= values_to_plot.shape[2]:
                    target = 0
                values_to_plot = values_to_plot[:, :, target]

        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            values_to_plot,
            X,
            feature_names=vectorizer.get_feature_names_out(),
            show=False,
            max_display=n_top
        )
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"SHAP summary plot saved to {save_path}")

        return fig

    except Exception as e:
        logger.warning(f"SHAP summary plot failed: {e}")
        return None
