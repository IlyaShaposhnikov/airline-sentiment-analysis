"""
Metrics computation, visualization, and export module.

Provides comprehensive evaluation metrics (accuracy, F1, ROC-AUC),
confusion matrix plotting, misclassified example extraction,
and flexible export to JSON/CSV. All functions auto-detect
binary vs multiclass tasks and handle edge cases gracefully.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils.logging_config import setup_logger

logger = setup_logger(__name__)


def _detect_task_type(y_true: np.ndarray) -> str:
    """
    Automatically detect classification task type based on unique labels.

    Args:
        y_true: Ground truth labels (1D array)

    Returns:
        'binary' if exactly 2 unique labels, else 'multiclass'
    """
    n_unique = len(np.unique(y_true))
    if n_unique == 2:
        logger.debug("Detected binary classification task")
        return "binary"
    else:
        logger.debug(
            f"Detected multiclass classification task ({n_unique} classes)"
        )
        return "multiclass"


def _compute_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    task_type: str | None = None,
    average: str = "macro",
) -> float | None:
    """
    Compute ROC-AUC score with automatic binary/multiclass handling.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities (shape: [n_samples, n_classes])
        task_type: Optional override ('binary' or 'multiclass')
        average: Averaging strategy for multiclass ('macro', 'weighted')

    Returns:
        ROC-AUC score or None if computation fails

    Note:
        - Binary: uses y_proba[:, 1] (positive class probability)
        - Multiclass: uses one-vs-one (ovo) strategy with macro averaging
        - Graceful degradation: returns None + warning on errors
    """
    if y_proba.size == 0 or y_proba.shape[1] == 0:
        logger.warning("Skipping AUC: empty or invalid probability array")
        return None

    if task_type is None:
        task_type = _detect_task_type(y_true)

    try:
        if task_type == "binary":
            # Binary AUC requires exactly 2 columns
            if y_proba.ndim != 2 or y_proba.shape[1] < 2:
                logger.warning(
                    "Binary AUC requires y_proba "
                    f"with shape [n, 2], got {y_proba.shape}"
                )
                return None
            return roc_auc_score(y_true, y_proba[:, 1])
        else:
            # Multiclass uses ovo strategy
            return roc_auc_score(
                y_true, y_proba, multi_class="ovo", average=average
            )
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC ({task_type}): {e}")
        return None


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list | None = None,
    tick_labels: list[str] | None = None,
    normalize: bool = True,
    cmap: str = "Blues",
    figsize: tuple[int, int] = (8, 6),
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot confusion matrix with optional normalization and custom labels.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Values to include in matrix (default: np.unique(y_true))
        tick_labels: String names for axis ticks (default: str(labels))
        normalize: If True, display rates instead of counts
        cmap: Matplotlib colormap name
        figsize: Figure size in inches (width, height)
        save_path: Optional path to save figure (creates parent dirs)
        title: Plot title

    Returns:
        Matplotlib Figure object

    Raises:
        ValueError: If tick_labels length doesn't match labels
    """
    if labels is None:
        labels = np.unique(y_true).tolist()

    if tick_labels is None:
        tick_labels = [str(label) for label in labels]
    elif len(tick_labels) != len(labels):
        raise ValueError(
            f"tick_labels length ({len(tick_labels)}) != "
            f"labels length ({len(labels)})"
        )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Safe normalization avoids division by zero
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # 0/0 → 0 (not NaN) for empty rows
        cm = np.divide(
            cm, row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        )
        fmt = ".2f"
        annot_label = "rate"
    else:
        fmt = "d"
        annot_label = "count"

    # Create DataFrame for seaborn heatmap
    df_cm = pd.DataFrame(cm, index=tick_labels, columns=tick_labels)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df_cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={"label": annot_label},
        ax=ax,
        square=True,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] | None = None,
    task_type: str | None = None,
    output_dict: bool = True,
) -> dict | pd.DataFrame:
    """
    Generate classification report with automatic binary/multiclass handling.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of class names for readable output
        task_type: Optional override ('binary' or 'multiclass')
        output_dict: If True, return dict; else return pandas DataFrame

    Returns:
        Classification report as dict or DataFrame

    Note:
        Uses sklearn's classification_report with zero_division=0
        to avoid warnings on empty class slices.
    """
    if task_type is None:
        task_type = _detect_task_type(y_true)

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    if output_dict:
        return report

    # Transpose for class-as-rows orientation
    df = pd.DataFrame(report).transpose()
    if "accuracy" in df.index:
        # Move accuracy to separate row for clarity
        acc_row = df.loc["accuracy"]
        df = df.drop("accuracy")
        df.loc["accuracy"] = acc_row
    return df


def get_top_misclassified(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    texts: list[str] | None = None,
    n_top: int = 10,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract top-N most confidently misclassified examples for debugging.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities [n_samples, n_classes]
        texts: Optional list of original text samples
        n_top: Number of top examples to return
        class_names: Optional list of class names for readable output

    Returns:
        DataFrame with misclassified examples sorted by prediction confidence

    Note:
        Confidence = probability assigned to the (wrong) predicted class.
        Higher confidence = more interesting error for model analysis.
    """
    misclassified_mask = y_true != y_pred
    if not np.any(misclassified_mask):
        logger.info("No misclassified examples found")
        return pd.DataFrame()

    # Extract data for misclassified samples only
    wrong_proba = y_proba[misclassified_mask]
    wrong_pred = y_pred[misclassified_mask]
    wrong_true = y_true[misclassified_mask]

    # Ensure integer arrays for safe indexing
    wrong_pred = np.asarray(wrong_pred, dtype=int)
    wrong_true = np.asarray(wrong_true, dtype=int)

    # Confidence = prob assigned to wrong prediction
    confidence = wrong_proba[np.arange(len(wrong_pred)), wrong_pred]

    # Sort by confidence descending (most confident errors first)
    top_indices = np.argsort(-confidence)[:n_top]

    results = []
    misclassified_indices = np.where(misclassified_mask)[0]

    for idx in top_indices:
        entry = {
            "true_label": int(wrong_true[idx]),
            "pred_label": int(wrong_pred[idx]),
            "confidence": float(confidence[idx]),
            # error_margin = how much model preferred wrong class
            "error_margin": float(
                confidence[idx] - wrong_proba[idx, wrong_true[idx]]
            ),
        }
        if class_names:
            entry["true_label_name"] = class_names[entry["true_label"]]
            entry["pred_label_name"] = class_names[entry["pred_label"]]
        if texts:
            entry["text"] = texts[misclassified_indices[idx]]
        results.append(entry)

    df = pd.DataFrame(results)
    logger.info(f"Extracted {len(df)} top misclassified examples")
    return df


def export_metrics(
    metrics: dict,
    output_dir: str | Path,
    filename: str = "metrics_report",
    formats: list[str] = ["json", "csv"],
) -> list[Path]:
    """
    Export metrics dictionary to JSON and/or CSV files.

    Args:
        metrics: Dictionary with metric names and values
        output_dir: Directory to save files (created if missing)
        filename: Base filename (without extension)
        formats: List of formats to export ('json', 'csv')

    Returns:
        List of paths to saved files

    Note:
        - JSON: full nested structure with numpy→Python conversion
        - CSV: flat metrics + optional classification_report table
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # Recursive helper for numpy→JSON serialization
    def _make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_make_serializable(i) for i in obj]
        return obj

    if "json" in formats:
        json_path = output_dir / f"{filename}.json"
        serializable = _make_serializable(metrics)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        saved_paths.append(json_path)
        logger.info(f"Metrics exported to {json_path}")

    if "csv" in formats:
        # Export only scalar metrics to flat CSV
        flat_metrics = {
            k: v for k, v in metrics.items()
            if not isinstance(v, (dict, list, np.ndarray))
        }

        if flat_metrics:
            csv_path = output_dir / f"{filename}_flat.csv"
            pd.DataFrame([flat_metrics]).T.to_csv(csv_path, header=["value"])
            saved_paths.append(csv_path)
            logger.info(f"Flat metrics exported to {csv_path}")

        # Export classification_report as separate table
        if "classification_report" in metrics and isinstance(
            metrics["classification_report"], dict
        ):
            report_csv_path = output_dir / f"{filename}_report.csv"
            try:
                report_df = (
                    pd.DataFrame(metrics["classification_report"]).transpose()
                )
                report_df.to_csv(report_csv_path)
                saved_paths.append(report_csv_path)
                logger.info(
                    f"Classification report exported to {report_csv_path}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not export classification report to CSV: {e}"
                )

    logger.debug(f"Exported metrics in formats: {formats} → {saved_paths}")
    return saved_paths


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    config: dict | None = None,
    class_names: list[str] | None = None,
    auto_export: bool = False,
    output_dir: str | Path | None = None,
    export_filename: str = "metrics_report",
) -> dict:
    """
    Compute comprehensive metrics with auto binary/multiclass detection.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities [n_samples, n_classes]
        config: Optional config dict with metric/reporting preferences
        class_names: Optional list of class names for reports
        auto_export: If True, automatically export metrics using config
        output_dir: Directory for exported files (required if auto_export=True)
        export_filename: Base filename for exported metrics

    Returns:
        Dictionary with all computed metrics

    Raises:
        ValueError: If input shapes are inconsistent

    Note:
        Always computes: accuracy, F1-macro, precision-macro, recall-macro
        Conditionally computes:
        ROC-AUC, confusion matrix, classification_report
        Auto-export uses config['evaluation']['reporting']['export_formats']
    """
    # Validate input shapes early for clear errors
    if not (len(y_true) == len(y_pred) == len(y_proba)):
        raise ValueError(
            f"Input length mismatch: "
            f"y_true={len(y_true)}, "
            f"y_pred={len(y_pred)}, "
            f"y_proba={len(y_proba)}"
        )
    if y_proba.ndim != 2:
        raise ValueError(
            f"y_proba must be 2D array, got shape {y_proba.shape}"
        )
    if y_proba.shape[0] != len(y_true):
        raise ValueError(
            f"y_proba rows ({y_proba.shape[0]}) != samples ({len(y_true)})"
        )

    config = config or {}
    eval_cfg = config.get("evaluation", {})
    reporting_cfg = eval_cfg.get("reporting", {})

    task_type = _detect_task_type(y_true)

    results = {
        "task_type": task_type,
        "n_samples": len(y_true),
        "n_classes": len(np.unique(y_true)),
    }

    # Baseline accuracy is always included
    results["accuracy"] = np.mean(y_true == y_pred)

    # Use 'binary' avg for binary, 'macro' for multiclass
    average = "binary" if task_type == "binary" else "macro"
    results["f1_macro"] = f1_score(
        y_true, y_pred, average=average, zero_division=0
    )
    results["precision_macro"] = precision_score(
        y_true, y_pred, average=average, zero_division=0
    )
    results["recall_macro"] = recall_score(
        y_true, y_pred, average=average, zero_division=0
    )

    # Weighted F1 only meaningful for multiclass
    if task_type == "multiclass":
        results["f1_weighted"] = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

    # ROC-AUC with graceful fallback
    auc_score = _compute_auc(y_true, y_proba, task_type=task_type)
    if auc_score is not None:
        results["roc_auc"] = auc_score

    # Confusion matrix as list for JSON serialization
    if reporting_cfg.get("include_confusion_matrix", True):
        unique_labels = np.unique(y_true).tolist()
        results["confusion_matrix"] = confusion_matrix(
            y_true, y_pred, labels=unique_labels
        ).tolist()

    # Detailed per-class metrics
    if reporting_cfg.get("include_classification_report", True):
        results["classification_report"] = generate_classification_report(
            y_true, y_pred, target_names=class_names, task_type=task_type
        )

    logger.info(
        f"Computed metrics: accuracy={results['accuracy']:.4f}, "
        f"f1={results['f1_macro']:.4f}, auc={results.get('roc_auc', 'N/A')}"
    )

    # Auto-export respects config['export_formats']
    if auto_export and output_dir:
        formats = reporting_cfg.get("export_formats", ["json", "csv"])
        logger.info(
            f"Auto-exporting metrics to {output_dir} in formats: {formats}"
        )
        try:
            export_metrics(
                results,
                output_dir=output_dir,
                filename=export_filename,
                formats=formats,
            )
        except Exception as e:
            logger.error(f"Failed to auto-export metrics: {e}")

    return results
