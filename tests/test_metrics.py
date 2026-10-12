"""
Regression tests for src/metrics.py metric definitions.
"""
import numpy as np
from sklearn.metrics import f1_score

from src.metrics import compute_comprehensive_metrics


def _binary_imbalanced():
    # 8 negatives, 2 positives; one positive missed, one false alarm
    y_true = np.array([0] * 8 + [1] * 2)
    y_pred = np.array([0] * 7 + [1] + [1, 0])
    proba = np.column_stack([1 - y_pred * 0.8, y_pred * 0.8])
    return y_true, y_pred, proba


def test_binary_f1_macro_is_macro_averaged():
    """
    Regression: binary runs used to store the positive-class F1 under
    "f1_macro" (0.5 here instead of the true macro 0.6875).
    """
    y_true, y_pred, proba = _binary_imbalanced()
    results = compute_comprehensive_metrics(y_true, y_pred, proba)

    expected = f1_score(y_true, y_pred, average="macro")
    assert results["f1_macro"] == expected
    assert results["f1_macro"] != f1_score(y_true, y_pred)


def test_binary_reports_positive_class_separately():
    y_true, y_pred, proba = _binary_imbalanced()
    results = compute_comprehensive_metrics(y_true, y_pred, proba)

    assert results["f1_positive"] == f1_score(y_true, y_pred)
    assert "precision_positive" in results
    assert "recall_positive" in results


def test_multiclass_has_no_positive_class_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    proba = np.eye(3)[y_pred] * 0.8 + 0.2 / 3
    results = compute_comprehensive_metrics(y_true, y_pred, proba)

    assert results["f1_macro"] == f1_score(y_true, y_pred, average="macro")
    assert "f1_positive" not in results
