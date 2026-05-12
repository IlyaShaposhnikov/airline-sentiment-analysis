from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .utils.logging_config import setup_logger

logger = setup_logger(__name__)


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
    vectorizer: Union[CountVectorizer, TfidfVectorizer],
    n_top: int = 20,
    threshold: Optional[float] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[str, float]]]:
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
        Dict mapping class name to list of (word, weight) tuples,
        with keys 'positive' and 'negative' for binary,
        or class names for multiclass.
    """
    # Get feature names from vectorizer
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions
        feature_names = np.array(vectorizer.get_feature_names())

    # Get model coefficients: shape [n_classes, n_features]
    coef = model.coef_
    classes = model.classes_

    results = {}

    for idx, class_label in enumerate(classes):
        # Use provided class name or fallback to label value
        class_key = (
            class_names[idx]
            if class_names and idx < len(class_names)
            else str(class_label)
        )

        # Get weights for this class
        weights = coef[idx]

        # Create (word, weight) pairs
        word_weights = list(zip(feature_names, weights))

        # Filter by threshold if specified
        if threshold is not None:
            word_weights = [
                (w, wt) for w, wt in word_weights if abs(wt) >= threshold
            ]

        # Sort by absolute weight
        word_weights_sorted = sorted(
            word_weights, key=lambda x: abs(x[1]), reverse=True
        )

        # Extract top positive and negative
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
    vectorizer: Union[CountVectorizer, TfidfVectorizer],
    class_idx: int = 0,
    n_top: int = 20,
    horizontal: bool = True,
    figsize: tuple = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot top features by weight for a specific class.

    Args:
        model: Trained LogisticRegression instance
        vectorizer: Fitted vectorizer
        class_idx: Index of class to visualize (0-based)
        n_top: Number of top features to show (positive + negative)
        horizontal: If True, plot horizontal bar chart (better for long words)
        figsize: Figure size in inches
        save_path: Optional path to save the figure
        title: Optional plot title (auto-generated if None)
        class_names: Optional list of class names for title/labels

    Returns:
        Matplotlib Figure object
    """
    # Get feature names and coefficients
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = np.array(vectorizer.get_feature_names())

    coef = model.coef_[class_idx]
    classes = model.classes_

    # Get class label for display
    class_label = (
        class_names[class_idx]
        if class_names and class_idx < len(class_names)
        else f"Class {classes[class_idx]}"
    )

    # Get top features by absolute weight
    top_indices = np.argsort(np.abs(coef))[-n_top:]
    top_features = feature_names[top_indices]
    top_weights = coef[top_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        # Horizontal bar chart - better for long words
        y_pos = np.arange(len(top_features))
        colors = ["green" if w > 0 else "red" for w in top_weights]
        ax.barh(y_pos, top_weights, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel("Weight")
        ax.set_title(title or f"Top {n_top} Features for '{class_label}'")
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    else:
        # Vertical bar chart
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

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")

    return fig


def _get_shap_explanation(
    model: LogisticRegression,
    vectorizer: Union[CountVectorizer, TfidfVectorizer],
    text: str,
    background_data: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """
    Internal function: compute SHAP explanation if SHAP is available.

    Returns None if SHAP is not installed or explanation fails.
    """
    if not SHAP_AVAILABLE:
        return None

    try:
        # Vectorize input text
        X_input = vectorizer.transform([text])

        # For LinearExplainer, we need background data for expected value
        # Use a small sample if not provided
        if background_data is None:
            # Fallback: use zeros as background (less accurate but works)
            logger.warning(
                "No background data for SHAP - using zero baseline. "
                "For better results, pass a sample of training data."
            )
            background_data = np.zeros((1, X_input.shape[1]))

        # Create explainer for linear model
        explainer = shap.LinearExplainer(
            model, background_data
        )

        # Compute SHAP values
        shap_values = explainer.shap_values(X_input)

        # Handle multiclass: shap_values is list of arrays
        # [n_classes][n_samples, n_features]
        if isinstance(shap_values, list):
            # Multiclass: return dict per class
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
            # Binary: single array
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
    vectorizer: Union[CountVectorizer, TfidfVectorizer],
    text: str,
    class_names: Optional[List[str]] = None,
    n_top_contributors: int = 10,
    use_shap: bool = False,
    background_data: Optional[np.ndarray] = None,
) -> Dict:
    """
    Explain a single prediction using model weights or SHAP values.

    Args:
        model: Trained LogisticRegression instance
        vectorizer: Fitted vectorizer
        text: Input text to explain
        class_names: Optional list of class names for readable output
        n_top_contributors: Number of top contributing words to return
        use_shap: If True and SHAP available, use SHAP values; else use weights
        background_data: Optional background data for SHAP (improves accuracy)

    Returns:
        Dict with prediction, probabilities, and top contributing features.
    """
    # Preprocess and vectorize input
    X_input = vectorizer.transform([text])

    # Get prediction and probabilities
    pred_idx = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    # Prepare class names mapping
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

    # Try SHAP if requested and available
    if use_shap and SHAP_AVAILABLE:
        shap_result = _get_shap_explanation(
            model, vectorizer, text, background_data
        )
        if shap_result is not None:
            result["method"] = "shap"
            # Use SHAP values for the predicted class
            if (
                isinstance(shap_result, dict)
                and f"class_{pred_idx}" in shap_result
            ):
                shap_vals = shap_result[f"class_{pred_idx}"]["values"]
            else:
                shap_vals = shap_result.get("values", None)

            if shap_vals is not None:
                # Get feature names and create (word, shap_value) pairs
                try:
                    feature_names = vectorizer.get_feature_names_out()
                except AttributeError:
                    feature_names = np.array(vectorizer.get_feature_names())

                # Only include non-zero contributions
                contributors = [
                    (feature_names[i], float(shap_vals[i]))
                    for i in range(len(shap_vals))
                    if shap_vals[i] != 0
                ]

                # Sort by absolute value and take top N
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

    # Fallback: use model weights (linear model interpretation)
    # Get weights for predicted class
    weights = model.coef_[pred_idx]

    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        feature_names = np.array(vectorizer.get_feature_names())

    # Get input feature indices (non-zero in transformed text)
    input_features = X_input.indices
    input_weights = weights[input_features]
    input_names = feature_names[input_features]

    # Create (word, weight) pairs for features present in input
    contributors = [
        (name, float(wt)) for name, wt in zip(input_names, input_weights)
        if wt != 0
    ]

    # Sort by absolute weight and take top N
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
    vectorizer: Union[CountVectorizer, TfidfVectorizer],
    texts: List[str],
    class_idx: Optional[int] = None,
    n_top: int = 20,
    save_path: Optional[Union[str, Path]] = None,
    random_state: Optional[int] = None,
) -> Optional[plt.Figure]:
    """
    Plot SHAP summary plot for multiple examples (if SHAP available).

    Args:
        model: Trained LogisticRegression instance
        vectorizer: Fitted vectorizer
        texts: List of input texts to explain
        class_idx: Optional class index to focus on (for multiclass)
        n_top: Number of top features to display
        save_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object or None if SHAP not available
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available - skipping summary plot")
        return None

    try:
        # Vectorize all texts
        X = vectorizer.transform(texts)

        # Create explainer
        # Use a subset of X as background for efficiency
        if random_state is not None:
            np.random.seed(random_state)
        background_idx = np.random.choice(
            len(texts), min(100, len(texts)), replace=False
        )
        background = X[background_idx]

        explainer = shap.LinearExplainer(
            model, background
        )

        # Compute SHAP values
        shap_values = explainer.shap_values(X)

        # Handle multiclass
        if isinstance(shap_values, list):
            if class_idx is not None and class_idx < len(shap_values):
                values_to_plot = shap_values[class_idx]
            else:
                # Plot for first class by default
                values_to_plot = shap_values[0]
        else:
            values_to_plot = shap_values

        # Create summary plot
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
