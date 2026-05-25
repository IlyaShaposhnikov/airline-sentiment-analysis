"""
Unit tests for src/api/models.py Pydantic v2 request/response schemas.

Focus: Input validation, boundary conditions, error handling,
and JSON schema generation.
"""

import pytest
from datetime import datetime, timezone

from pydantic import Field, ValidationError
from unittest.mock import patch

from src.api.models import (
    BaseAPIModel,
    Explanation,
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
)
from src.api.config import (
    MAX_TEXT_LENGTH,
    MAX_BATCH_SIZE,
    MIN_EXPLAIN_COUNT,
    MAX_EXPLAIN_COUNT,
)


# ============================================================================
# Base model tests
# ============================================================================


class TestBaseAPIModel:
    """Tests for BaseAPIModel configuration."""

    def test_str_strip_whitespace_enabled(self):
        """BaseAPIModel should strip whitespace from string fields."""
        class TestModel(BaseAPIModel):
            name: str

        # Whitespace should be stripped automatically
        instance = TestModel(name="  test  ")
        assert instance.name == "test"

    def test_validate_default_enabled(self):
        """BaseAPIModel should validate default values."""
        class TestModel(BaseAPIModel):
            value: int = Field(default=5, ge=0, le=10)

        # Default value should pass validation
        instance = TestModel()
        assert instance.value == 5

    def test_child_models_inherit_base_config(self):
        """Verify str_strip_whitespace and validate_default are inherited."""
        assert PredictionRequest.model_config.get(
            "str_strip_whitespace"
        ) is True
        assert PredictionRequest.model_config.get("validate_default") is True
        assert BatchPredictionRequest.model_config.get(
            "str_strip_whitespace"
        ) is True
        assert BatchPredictionResponse.model_config.get(
            "validate_default"
        ) is True

# ============================================================================
# Explanation model tests
# ============================================================================


class TestExplanation:
    """Tests for the Explanation schema."""

    def test_valid_explanation(self):
        """Valid explanation should instantiate correctly."""
        exp = Explanation(
            method="weights",
            top_contributors=[("great", 1.5), ("flight", 0.8)],
        )
        assert exp.method == "weights"
        assert len(exp.top_contributors) == 2
        assert exp.top_contributors[0] == ("great", 1.5)

    def test_method_literal_validation(self):
        """Method field should only accept 'weights' or 'shap'."""
        # Valid values
        Explanation(method="weights", top_contributors=[("test", 1.0)])
        Explanation(method="shap", top_contributors=[("test", 1.0)])

        # Invalid value should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            Explanation(method="invalid", top_contributors=[("test", 1.0)])

        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in ["literal", "input"]), (
            "Expected literal validation error for method='invalid', "
            f"got: {exc_info.value}"
        )

    def test_top_contributors_structure(self):
        """top_contributors should be list of (str, float) tuples."""
        # Valid: Pydantic v2 coerces lists to tuples automatically
        Explanation(
            method="weights", top_contributors=[
                ("word", 1.0), ["another", -0.5]  # list is coerced to tuple
            ]
        )

        # Invalid: wrong number of elements in tuple (expected 2, got 3)
        with pytest.raises(ValidationError) as exc_info:
            Explanation(
                method="weights", top_contributors=[("word", 1.0, "extra")]
            )
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "tuple has wrong number of elements", "unexpected", "3"
        ]), f"Expected tuple length error, got: {exc_info.value}"

    def test_json_schema_extra_example(self):
        """Explanation should include example in JSON schema."""
        schema = Explanation.model_json_schema()
        assert "examples" in schema
        examples = schema["examples"]
        assert len(examples) > 0
        assert examples[0]["method"] == "weights"
        assert "top_contributors" in examples[0]

    def test_empty_top_contributors_allowed(self):
        """Empty contributors list should be allowed (valid edge case)."""
        exp = Explanation(method="weights", top_contributors=[])
        assert exp.top_contributors == []

    def test_top_contributors_reject_invalid_types(self):
        """Non-coercible values in contributors should fail validation."""
        # String that cannot be coerced to float should fail
        with pytest.raises(ValidationError) as exc_info:
            Explanation(
                method="weights", top_contributors=[("word", "not-a-float")]
            )
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "input should be a valid number", "cannot be parsed", "float"
        ]), f"Expected float coercion error, got: {exc_info.value}"

        # None value should fail (cannot coerce None to float)
        with pytest.raises(ValidationError) as exc_info:
            Explanation(
                method="weights", top_contributors=[("word", None)]
            )
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "input should be a valid number", "none", "float"
        ]), f"Expected None rejection error, got: {exc_info.value}"


# ============================================================================
# PredictionRequest tests
# ============================================================================


class TestPredictionRequest:
    """Tests for the PredictionRequest schema."""

    def test_minimal_valid_request(self, api_valid_prediction_request_data):
        """Minimal valid request should instantiate."""
        req = PredictionRequest(**api_valid_prediction_request_data)
        assert req.text == "Great flight!"
        assert req.explain is False  # Default value
        assert req.use_shap is False  # Default value
        assert req.n_explain == MIN_EXPLAIN_COUNT  # Default value

    def test_text_length_validation(self):
        """
        Text field should enforce min_length=1 and max_length=MAX_TEXT_LENGTH.
        """
        # Empty string should fail
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(text="")
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "min_length", "at least 1 character", "string too short"
        ]), f"Expected min_length error for empty text, got: {exc_info.value}"

        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(text="   ")
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "min_length", "at least 1 character", "string too short"
        ]), f"Expected min_length error, got: {exc_info.value}"

        # Text at max length should pass
        valid_long = "a" * MAX_TEXT_LENGTH
        req = PredictionRequest(text=valid_long)
        assert len(req.text) == MAX_TEXT_LENGTH

        # Text exceeding max length should fail
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(text="a" * (MAX_TEXT_LENGTH + 1))
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "max_length", "at most", "string too long"
        ]), (
            f"Expected max_length error for text > {MAX_TEXT_LENGTH}, "
            f"got: {exc_info.value}"
        )

    def test_n_explain_validation(self):
        """
        n_explain should be constrained to
        [MIN_EXPLAIN_COUNT, MAX_EXPLAIN_COUNT].
        """
        # Boundary values should pass
        PredictionRequest(text="test", n_explain=MIN_EXPLAIN_COUNT)
        PredictionRequest(text="test", n_explain=MAX_EXPLAIN_COUNT)

        # Below minimum should fail
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(text="test", n_explain=MIN_EXPLAIN_COUNT - 1)
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "greater than or equal to", "ge", "input should be"
        ]), (
            f"Expected ge validation error for n_explain < {MIN_EXPLAIN_COUNT}"
            f", got: {exc_info.value}"
        )

        # Above maximum should fail
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(text="test", n_explain=MAX_EXPLAIN_COUNT + 1)
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "less than or equal to", "le", "input should be"
        ]), (
            f"Expected le validation error for n_explain > {MAX_EXPLAIN_COUNT}"
            f", got: {exc_info.value}"
        )

    def test_optional_boolean_fields(self, api_valid_prediction_request_data):
        """explain and use_shap should default to False."""
        # Defaults
        req = PredictionRequest(**api_valid_prediction_request_data)
        assert req.explain is False
        assert req.use_shap is False

        # Explicit True values
        req = PredictionRequest(text="test", explain=True, use_shap=True)
        assert req.explain is True
        assert req.use_shap is True

    def test_json_schema_examples(self):
        """PredictionRequest should include examples in JSON schema."""
        schema = PredictionRequest.model_json_schema()
        assert "examples" in schema["properties"]["text"]
        examples = schema["properties"]["text"]["examples"]
        assert len(examples) >= 2
        assert any("Great flight" in ex for ex in examples)


# ============================================================================
# BatchPredictionRequest tests
# ============================================================================


class TestBatchPredictionRequest:
    """Tests for the BatchPredictionRequest schema."""

    def test_minimal_valid_batch_request(self, api_valid_batch_request_data):
        """Minimal valid batch request should instantiate."""
        req = BatchPredictionRequest(**api_valid_batch_request_data)
        assert len(req.texts) == 2
        assert req.texts[0] == "Great flight!"
        assert req.explain is False

    def test_texts_list_validation(self):
        """texts field should be a non-empty list of strings."""
        # Empty list should fail (min_length=1 on the list itself)
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest(texts=[])
        # Pydantic v2 error message for min_length on list
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "at least 1 item", "min_length", "list should have"
        ]), (
            "Expected min_length error for empty texts list, "
            f"got: {exc_info.value}"
        )

        # List with valid strings should pass
        req = BatchPredictionRequest(texts=["a", "b", "c"])
        assert len(req.texts) == 3

    def test_custom_validator_empty_text_in_list(self):
        """
        validate_texts should reject empty
        or whitespace-only strings in the list.
        """
        # Empty string in list
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest(texts=["valid", "", "also valid"])
        assert "cannot be empty or whitespace-only" in str(exc_info.value)

        # Whitespace-only string in list
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest(texts=["valid", "   ", "also valid"])
        assert "cannot be empty or whitespace-only" in str(exc_info.value)

    def test_custom_validator_text_length_in_list(self):
        """validate_texts should reject texts exceeding MAX_TEXT_LENGTH."""
        too_long = "a" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest(texts=["valid", too_long])
        assert (
            f"exceeds max length of {MAX_TEXT_LENGTH}" in str(exc_info.value)
        )

    def test_batch_size_limit(self):
        """texts list should not exceed MAX_BATCH_SIZE items."""
        # At limit should pass
        valid_batch = ["text"] * MAX_BATCH_SIZE
        req = BatchPredictionRequest(texts=valid_batch)
        assert len(req.texts) == MAX_BATCH_SIZE

        # Exceeding limit should fail
        too_many = ["text"] * (MAX_BATCH_SIZE + 1)
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionRequest(texts=too_many)
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "at most", "max_length", "list should have"
        ]), (
            f"Expected max_length error for texts list > {MAX_BATCH_SIZE}, "
            f"got: {exc_info.value}"
        )

    def test_inheritance_from_base_prediction_request(self):
        """BatchPredictionRequest should inherit n_explain constraints."""
        # n_explain should still be validated
        with pytest.raises(ValidationError):
            BatchPredictionRequest(
                texts=["test"], n_explain=MIN_EXPLAIN_COUNT - 1
            )

        req = BatchPredictionRequest(texts=["test"], n_explain=10)
        assert req.n_explain == 10


# ============================================================================
# PredictionResponse tests
# ============================================================================


class TestPredictionResponse:
    """Tests for the PredictionResponse schema."""

    def test_minimal_valid_response(self, api_valid_prediction_response_data):
        """Minimal valid response should instantiate."""
        resp = PredictionResponse(**api_valid_prediction_response_data)
        assert resp.text == "Great flight!"
        assert resp.predicted_class == "positive"
        assert resp.confidence == 0.8
        assert isinstance(resp.timestamp, datetime)
        assert resp.timestamp.tzinfo == timezone.utc  # Should be UTC

    def test_confidence_range_validation(self):
        """confidence should be constrained to [0.0, 1.0]."""
        # Boundary values should pass
        PredictionResponse(
            text="test", predicted_class="positive", predicted_class_idx=1,
            probabilities={"pos": 1.0}, confidence=0.0
        )
        PredictionResponse(
            text="test", predicted_class="positive", predicted_class_idx=1,
            probabilities={"pos": 1.0}, confidence=1.0
        )

        # Below minimum should fail
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(
                text="test", predicted_class="positive", predicted_class_idx=1,
                probabilities={"pos": 1.0}, confidence=-0.1
            )
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "greater than or equal to 0", "ge", "input should be"
        ]), f"Expected confidence >= 0 error, got: {exc_info.value}"

        # Above maximum should fail
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(
                text="test", predicted_class="positive", predicted_class_idx=1,
                probabilities={"pos": 1.0}, confidence=1.1
            )
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "less than or equal to 1", "le", "input should be"
        ]), f"Expected confidence <= 1 error, got: {exc_info.value}"

    def test_probabilities_dict_structure(self):
        """probabilities should be a dict with string keys and float values."""
        # Valid structure
        resp = PredictionResponse(
            text="test", predicted_class="positive", predicted_class_idx=1,
            probabilities={"negative": 0.1, "positive": 0.8, "neutral": 0.1},
            confidence=0.8
        )
        assert isinstance(resp.probabilities, dict)
        assert all(isinstance(k, str) for k in resp.probabilities.keys())
        assert all(isinstance(v, float) for v in resp.probabilities.values())

    def test_optional_explanation_field(
            self, api_valid_prediction_response_data
    ):
        """explanation field should be optional."""
        # Without explanation
        resp = PredictionResponse(**api_valid_prediction_response_data)
        assert resp.explanation is None

        # With explanation
        data = api_valid_prediction_response_data.copy()
        data["explanation"] = {
            "method": "weights",
            "top_contributors": [("great", 1.5)]
        }
        resp = PredictionResponse(**data)
        assert resp.explanation is not None
        assert resp.explanation.method == "weights"

    @patch("src.api.models.datetime")
    def test_timestamp_uses_utc_now(self, mock_datetime):
        """
        timestamp should use datetime.now(timezone.utc) — tested with mock.
        """
        fixed = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = fixed
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)

        resp = PredictionResponse(
            text="test", predicted_class="positive", predicted_class_idx=1,
            probabilities={"pos": 1.0}, confidence=1.0
        )

        assert resp.timestamp == fixed
        assert resp.timestamp.tzinfo == timezone.utc

    def test_json_schema_extra_example(self):
        """
        PredictionResponse should include comprehensive example in JSON schema.
        """
        schema = PredictionResponse.model_json_schema()
        assert "examples" in schema
        example = schema["examples"][0]
        assert example["predicted_class"] == "positive"
        assert "probabilities" in example
        assert "explanation" in example

    def test_probabilities_schema_allows_negative_values(self):
        """
        Document that schema doesn't enforce [0,1] on individual probabilities.
        Business logic layer should validate this separately.
        """
        resp = PredictionResponse(
            text="test", predicted_class="positive", predicted_class_idx=1,
            probabilities={"negative": -0.1, "positive": 1.0},
            confidence=0.8
        )
        assert resp.probabilities["negative"] == -0.1


# ============================================================================
# BatchPredictionResponse tests
# ============================================================================


class TestBatchPredictionResponse:
    """Tests for the BatchPredictionResponse schema."""

    def test_minimal_valid_batch_response(self):
        """Minimal valid batch response should instantiate."""
        resp = BatchPredictionResponse(
            count=1,
            predictions=[
                {
                    "text": "test",
                    "predicted_class": "positive",
                    "predicted_class_idx": 1,
                    "probabilities": {"pos": 1.0},
                    "confidence": 1.0,
                }
            ],
            processing_time_ms=45.2,
        )
        assert resp.count == 1
        assert len(resp.predictions) == 1
        assert resp.status == "success"  # Default value

    def test_count_consistency_validator(self):
        """
        validate_count_consistency should ensure
        count matches predictions length.
        """
        # Matching count should pass
        BatchPredictionResponse(
            count=2,
            predictions=[
                {
                    "text": "a",
                    "predicted_class": "pos",
                    "predicted_class_idx": 1,
                    "probabilities": {"p": 1.0},
                    "confidence": 1.0
                },
                {
                    "text": "b",
                    "predicted_class": "pos",
                    "predicted_class_idx": 1,
                    "probabilities": {"p": 1.0},
                    "confidence": 1.0
                },
            ],
            processing_time_ms=10.0,
        )

        # Mismatched count should fail
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionResponse(
                count=1,  # Says 1
                predictions=[
                    {
                        "text": "a",
                        "predicted_class": "pos",
                        "predicted_class_idx": 1,
                        "probabilities": {"p": 1.0},
                        "confidence": 1.0
                    },
                    {
                        "text": "b",
                        "predicted_class": "pos",
                        "predicted_class_idx": 1,
                        "probabilities": {"p": 1.0},
                        "confidence": 1.0
                    },
                ],  # But has 2
                processing_time_ms=10.0,
            )
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "does not match", "count", "predictions length"
        ]), f"Expected count mismatch error, got: {exc_info.value}"

    def test_processing_time_non_negative(self):
        """processing_time_ms should be >= 0.0."""
        # Zero should pass
        BatchPredictionResponse(
            count=0, predictions=[], processing_time_ms=0.0
        )

        # Negative should fail
        with pytest.raises(ValidationError) as exc_info:
            BatchPredictionResponse(
                count=0, predictions=[], processing_time_ms=-1.0
            )
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "greater than or equal to 0", "ge", "input should be"
        ]), f"Expected processing_time >= 0 error, got: {exc_info.value}"


# ============================================================================
# HealthResponse tests
# ============================================================================


class TestHealthResponse:
    """Tests for the HealthResponse schema."""

    def test_minimal_valid_health_response(self):
        """Minimal valid health response should instantiate."""
        resp = HealthResponse(
            model_loaded=True,
            shap_available=False,
        )
        assert resp.status == "healthy"  # Default
        assert resp.service == "airline-sentiment-api"  # Default
        assert isinstance(resp.timestamp, datetime)
        assert resp.timestamp.tzinfo == timezone.utc

    def test_required_boolean_fields(self):
        """model_loaded and shap_available are required (no defaults)."""
        # Both provided - should pass
        HealthResponse(model_loaded=True, shap_available=True)
        HealthResponse(model_loaded=False, shap_available=False)

        # Missing one should fail
        with pytest.raises(ValidationError) as exc_info:
            HealthResponse(model_loaded=True)  # missing shap_available
        error_msg = str(exc_info.value).lower()
        assert any(kw in error_msg for kw in [
            "field required", "shap_available", "missing"
        ]), (
            "Expected 'field required' error for missing shap_available, "
            f"got: {exc_info.value}"
        )


# ============================================================================
# ErrorResponse tests
# ============================================================================


class TestErrorResponse:
    """Tests for the ErrorResponse schema."""

    def test_minimal_valid_error_response(self):
        """Minimal valid error response should instantiate."""
        resp = ErrorResponse(
            error="validation_error",
            detail="Request validation failed",
        )
        assert resp.timestamp.tzinfo == timezone.utc

    def test_required_string_fields(self):
        """error and detail are required fields."""
        # Both provided - should pass
        ErrorResponse(error="test", detail="description")

        # Missing one should fail
        with pytest.raises(ValidationError):
            ErrorResponse(error="test")  # missing detail


# ============================================================================
# Integration: JSON serialization/deserialization
# ============================================================================


class TestJSONSerialization:
    """Tests for JSON (de)serialization of API models."""

    def test_prediction_request_roundtrip(
            self, api_valid_prediction_request_data
    ):
        """PredictionRequest should serialize and deserialize correctly."""
        # Create instance
        req = PredictionRequest(**api_valid_prediction_request_data)

        # Serialize to JSON
        json_str = req.model_dump_json()

        # Deserialize back
        req2 = PredictionRequest.model_validate_json(json_str)

        # Should be equal
        assert req.text == req2.text
        assert req.explain == req2.explain
        assert req.n_explain == req2.n_explain

    def test_prediction_response_with_explanation_roundtrip(self):
        """PredictionResponse with explanation should roundtrip correctly."""
        resp = PredictionResponse(
            text="test",
            predicted_class="positive",
            predicted_class_idx=1,
            probabilities={"negative": 0.1, "positive": 0.8, "neutral": 0.1},
            confidence=0.8,
            explanation={
                "method": "weights",
                "top_contributors": [("great", 1.5), ("flight", 0.8)],
            },
        )

        # Roundtrip
        json_str = resp.model_dump_json()
        resp2 = PredictionResponse.model_validate_json(json_str)

        assert resp2.explanation is not None
        assert resp2.explanation.method == "weights"
        assert len(resp2.explanation.top_contributors) == 2

    def test_batch_response_json_schema(self):
        """
        BatchPredictionResponse JSON schema should be valid
        and include examples.
        """
        schema = BatchPredictionResponse.model_json_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "count" in schema["properties"]
        assert "predictions" in schema["properties"]
        assert "examples" in schema
