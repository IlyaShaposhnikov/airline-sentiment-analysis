"""
Model interpretability module for explaining predictions.

Provides weight-based feature attribution for LogisticRegression,
optional SHAP integration for advanced explanations, and visualization
utilities. All functions gracefully degrade if SHAP is unavailable.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Optional dependency pattern with graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
    logger.debug("SHAP library available - enabling advanced explanations")
except ImportError:
    SHAP_AVAILABLE = False
    logger.debug(
        "SHAP library not available - using weight-based explanations only"
    )
    shap = None  # type: ignore


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

    coef = model.coef_[class_idx]
    classes = model.classes_

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
) -> dict | None:
    """
    Internal: compute SHAP explanation if SHAP is available.

    Args:
        model: Trained LogisticRegression instance
        vectorizer: Fitted vectorizer
        text: Input text to explain
        background_data: Optional background data for expected value estimation

    Returns:
        Dict with SHAP values per class or None if unavailable/failed

    Note:
        Uses LinearExplainer for efficient exact SHAP values on linear models.
        Falls back to zero baseline if no background data provided
        (less accurate).
    """
    if not SHAP_AVAILABLE:
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

        # Multiclass returns list[class_idx][sample, feature]
        if isinstance(shap_values, list):
            result = {}
            for idx, class_shap in enumerate(shap_values):
                try:
                    base_val = explainer.expected_value[idx]
                except (AttributeError, IndexError, TypeError, KeyError):
                    base_val = 0.0

                result[f"class_{idx}"] = {
                    "values": class_shap[0],
                    "base_value": base_val,
                }
            return result
        else:
            try:
                base_val = explainer.expected_value
            except (AttributeError, TypeError, KeyError):
                base_val = 0.0
            return {
                "values": shap_values[0],
                "base_value": base_val,
            }

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
        shap_result = _get_shap_explanation(
            model, vectorizer, text, background_data
        )
        if shap_result is not None:
            result["method"] = "shap"
            if (
                isinstance(shap_result, dict)
                and f"class_{pred_idx}" in shap_result
            ):
                shap_vals = shap_result[f"class_{pred_idx}"]["values"]
            else:
                shap_vals = shap_result.get("values", None)

            if shap_vals is not None:
                try:
                    feature_names = vectorizer.get_feature_names_out()
                except AttributeError:
                    feature_names = np.array(vectorizer.get_feature_names())

                # Handle potential length mismatch
                # between SHAP output and features
                shap_vals_flat = np.asarray(shap_vals).flatten()
                n_features = min(len(feature_names), len(shap_vals_flat))
                if len(shap_vals_flat) != len(feature_names):
                    logger.debug(
                        f"SHAP values length ({len(shap_vals_flat)}) != "
                        f"feature_names length ({len(feature_names)}). "
                        f"Truncating to {n_features} features."
                    )

                shap_vals_flat = shap_vals_flat[:n_features]
                feature_names = feature_names[:n_features]

                nonzero_mask = np.abs(shap_vals_flat) > 1e-10
                nonzero_indices = np.where(nonzero_mask)[0]

                contributors = [
                    (feature_names[i], float(shap_vals_flat[i]))
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
    if not SHAP_AVAILABLE:
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

        # Multiclass SHAP returns list; select target class
        if isinstance(shap_values, list):
            if class_idx is not None and class_idx < len(shap_values):
                values_to_plot = shap_values[class_idx]
            else:
                values_to_plot = shap_values[0]
        else:
            values_to_plot = shap_values

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
