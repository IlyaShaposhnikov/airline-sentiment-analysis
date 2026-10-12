"""
SHAP explanations map values to the right words and class.

shap >= 0.45 returns multiclass values as one array of shape
(n_samples, n_features, n_classes) instead of a list per class. Flattening
that array pairs values with the wrong words, so the output format is
normalized before use. A fake ``shap`` module computes exact linear SHAP
values (x - background) * coef in each format, so these tests need no shap.
"""
import types

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import interpretability
from src.interpretability import _shap_values_for_class, explain_prediction

TEXTS = [
    "great flight thanks", "awesome crew great", "love this airline",
    "terrible delay again", "lost my bag awful", "worst service delay",
    "flight at noon", "gate number change", "what time boarding",
] * 3
LABELS_3 = [1, 1, 1, 0, 0, 0, 2, 2, 2] * 3
LABELS_2 = [1, 1, 1, 0, 0, 0, 1, 0, 1] * 3


def _fit(labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(TEXTS)
    model = LogisticRegression(max_iter=500).fit(X, labels)
    return model, vectorizer


def _fake_shap(output_format: str):
    """Fake shap module whose LinearExplainer returns exact linear SHAP."""

    class LinearExplainer:
        def __init__(self, model, data):
            self.coef = model.coef_
            self.mean = np.asarray(data.mean(axis=0)).ravel()
            self.expected_value = (
                self.coef @ self.mean + model.intercept_
            )

        def shap_values(self, X):
            x = np.asarray(X.toarray() if hasattr(X, "toarray") else X)
            per_class = [(x - self.mean) * c for c in self.coef]
            if len(per_class) == 1:
                return per_class[0]  # binary LR: single output
            if output_format == "list":
                return per_class  # shap < 0.45
            return np.stack(per_class, axis=-1)  # shap >= 0.45

    return types.SimpleNamespace(LinearExplainer=LinearExplainer)


@pytest.fixture
def use_fake_shap(monkeypatch):
    def install(output_format):
        monkeypatch.setattr(interpretability, "SHAP_AVAILABLE", True)
        monkeypatch.setattr(
            interpretability, "_shap_module", _fake_shap(output_format)
        )
    return install


def _expected(model, vectorizer, text, class_pos):
    x = vectorizer.transform([text]).toarray()[0]
    if model.coef_.shape[0] == 1:
        coef = model.coef_[0] if class_pos == 1 else -model.coef_[0]
    else:
        coef = model.coef_[class_pos]
    names = vectorizer.get_feature_names_out()
    return {
        names[i]: x[i] * coef[i] for i in np.flatnonzero(x * coef)
    }


class TestShapValuesForClass:
    def test_list_format(self):
        values = [np.full((1, 4), k, dtype=float) for k in range(3)]
        np.testing.assert_array_equal(
            _shap_values_for_class(values, 2, 3), np.full(4, 2.0)
        )

    def test_trailing_class_axis_format(self):
        values = np.arange(12, dtype=float).reshape(1, 4, 3)
        np.testing.assert_array_equal(
            _shap_values_for_class(values, 1, 3), [1, 4, 7, 10]
        )

    def test_single_output_negated_for_class_0(self):
        values = np.array([[0.5, -1.0]])
        np.testing.assert_array_equal(
            _shap_values_for_class(values, 1, 2), [0.5, -1.0]
        )
        np.testing.assert_array_equal(
            _shap_values_for_class(values, 0, 2), [-0.5, 1.0]
        )


@pytest.mark.parametrize("output_format", ["list", "array"])
def test_multiclass_shap_pairs_values_with_right_words(
    use_fake_shap, output_format
):
    use_fake_shap(output_format)
    model, vectorizer = _fit(LABELS_3)
    text = "terrible delay great crew"

    result = explain_prediction(
        model, vectorizer, text, use_shap=True, n_top_contributors=50
    )

    assert result["method"] == "shap"
    class_pos = list(model.classes_).index(result["predicted_class_idx"])
    expected = _expected(model, vectorizer, text, class_pos)
    got = dict(result["top_contributors"])
    assert got.keys() == expected.keys()
    for word, value in expected.items():
        assert got[word] == pytest.approx(value)


@pytest.mark.parametrize("text", ["great flight thanks", "terrible delay"])
def test_binary_shap_matches_weight_sign_convention(use_fake_shap, text):
    use_fake_shap("array")
    model, vectorizer = _fit(LABELS_2)

    shap_result = explain_prediction(model, vectorizer, text, use_shap=True)
    weight_result = explain_prediction(
        model, vectorizer, text, use_shap=False
    )

    assert shap_result["method"] == "shap"
    shap_words = dict(shap_result["top_contributors"])
    weight_words = dict(weight_result["top_contributors"])
    # Zero baseline: SHAP = x * weight, so signs agree word by word
    for word, weight in weight_words.items():
        assert np.sign(shap_words[word]) == np.sign(weight)


def test_mismatched_shap_length_falls_back_to_weights(
    use_fake_shap, monkeypatch
):
    use_fake_shap("array")
    model, vectorizer = _fit(LABELS_3)
    monkeypatch.setattr(
        interpretability,
        "_get_shap_explanation",
        lambda *a, **k: {"values": np.ones(3), "base_value": 0.0},
    )

    result = explain_prediction(
        model, vectorizer, "terrible delay", use_shap=True
    )

    assert result["method"] == "weights"
    assert result["top_contributors"]
