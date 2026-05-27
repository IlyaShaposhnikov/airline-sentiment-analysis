"""
Pydantic v2 models for API request/response validation.

Defines strict contracts for incoming requests and outgoing responses.
All input/output data is validated against these schemas before processing.
"""
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import (
    BaseModel, Field, field_validator, model_validator, ConfigDict
)

from .config import (
    MAX_BATCH_SIZE,
    MAX_EXPLAIN_COUNT,
    MAX_TEXT_LENGTH,
    MIN_EXPLAIN_COUNT,
)


# ============================================================================
# Base models & Config
# ============================================================================


class BaseAPIModel(BaseModel):
    """Base config for all API models."""
    model_config = ConfigDict(
        str_strip_whitespace=True,  # Auto-strip leading/trailing whitespace
        validate_default=True,  # Ensure default values pass constraints
    )


class BasePredictionRequest(BaseAPIModel):
    """Shared fields for prediction requests."""
    n_explain: Annotated[
        int,
        Field(
            default=MIN_EXPLAIN_COUNT,
            ge=MIN_EXPLAIN_COUNT,
            le=MAX_EXPLAIN_COUNT,
            description="Number of top contributing words to return (1-20)"
        )
    ]


class Explanation(BaseAPIModel):
    """Typed explanation schema for word-level contribution scores."""
    method: Literal["weights", "shap"] = Field(
        ..., description="Explanation method used"
    )
    # Pydantic v2 auto-coerces incoming JSON lists to tuples for type safety
    top_contributors: list[tuple[str, float]] = Field(
        ..., description="List of (word, contribution_score) pairs"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "method": "weights",
                "top_contributors": [["great", 1.85], ["flight", 0.45]]
            }]
        }
    )

# ============================================================================
# Request models
# ============================================================================


class PredictionRequest(BasePredictionRequest):
    """Request model for single text sentiment prediction."""
    text: Annotated[
        str,
        Field(
            ...,
            min_length=1,
            max_length=MAX_TEXT_LENGTH,
            description="Text to analyze for sentiment",
            examples=[
                "Great flight, excellent service!",
                "Terrible delay never again"
            ]
        )
    ]
    explain: Annotated[
        bool,
        Field(
            False,
            description="Include explanation with top contributing words"
        )
    ] = False
    use_shap: Annotated[
        bool,
        Field(
            False,
            description="Use SHAP for explanation (requires shap package)"
        )
    ] = False


class BatchPredictionRequest(BasePredictionRequest):
    """Request model for batch sentiment prediction."""
    texts: Annotated[
        list[str],
        Field(
            ...,
            min_length=1,
            max_length=MAX_BATCH_SIZE,
            description="List of 1-100 texts to analyze",
            examples=[["Great flight!", "Terrible service", "Okay experience"]]
        )
    ]
    explain: Annotated[
        bool,
        Field(False, description="Include explanations for all predictions")
    ] = False
    use_shap: Annotated[
        bool,
        Field(
            False,
            description="Use SHAP for explanations (requires shap package)"
        )
    ] = False

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        # Handles list length,
        # but per-item constraints require custom validator
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(
                    f"Text at index {i} cannot be empty or whitespace-only"
                )
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"Text at index {i} exceeds "
                    f"max length of {MAX_TEXT_LENGTH}"
                )
        return v

# ============================================================================
# Response models
# ============================================================================


class PredictionResponse(BaseAPIModel):
    """Response model for single text sentiment prediction."""
    text: str = Field(..., description="Original input text")
    predicted_class: str = Field(..., description="Predicted sentiment label")
    predicted_class_idx: int = Field(..., description="Numeric class index")
    probabilities: dict[str, float] = Field(
        ..., description="Probability scores for each class"
    )
    confidence: Annotated[float, Field(
        ..., ge=0.0, le=1.0, description="Model confidence")]
    # default_factory with lambda
    # ensures each response gets a fresh UTC timestamp
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of prediction"
    )
    explanation: Explanation | None = Field(
        None, description="Optional explanation with top contributing words"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "text": "Great flight!",
                "predicted_class": "positive",
                "predicted_class_idx": 1,
                "probabilities": {
                    "negative": 0.02, "positive": 0.96, "neutral": 0.02
                },
                "confidence": 0.96,
                "explanation": {
                    "method": "weights",
                    "top_contributors": [["great", 1.85], ["flight", 0.45]]
                }
            }]
        }
    )


class BatchPredictionResponse(BaseAPIModel):
    """Response model for batch sentiment prediction."""
    status: Annotated[
        str, Field("success", description="Overall status")
    ] = "success"
    count: Annotated[int, Field(
        ..., ge=0, description="Number of predictions"
    )]
    predictions: list[PredictionResponse] = Field(
        ..., description="List of results"
    )
    processing_time_ms: Annotated[float, Field(
        ..., ge=0.0, description="Total processing time in milliseconds"
    )]

    @model_validator(mode='after')
    def validate_count_consistency(self) -> 'BatchPredictionResponse':
        # Cross-field validation:
        # ensures declared count matches actual list length
        if self.count != len(self.predictions):
            raise ValueError(
                f"count ({self.count}) does not match "
                f"predictions length ({len(self.predictions)})"
            )
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "status": "success",
                "count": 1,
                "predictions": [{
                    "text": "Great flight!",
                    "predicted_class": "positive",
                    "predicted_class_idx": 1,
                    "probabilities": {
                        "negative": 0.02, "positive": 0.96, "neutral": 0.02
                    },
                    "confidence": 0.96,
                    "timestamp": "2026-05-18T10:00:00.123456"
                }],
                "processing_time_ms": 45.2
            }]
        }
    )


class HealthResponse(BaseAPIModel):
    """Response model for health check endpoint."""
    status: Annotated[str, Field(
        "healthy", description="Health status"
    )] = "healthy"
    service: Annotated[str, Field(
        "airline-sentiment-api", description="Service name"
    )] = "airline-sentiment-api"
    # Fresh timestamp generated on each instantiation
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Current UTC timestamp"
    )
    model_loaded: bool = Field(..., description="Whether model is loaded")
    shap_available: bool = Field(..., description="Whether SHAP is available")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "status": "healthy",
                "service": "airline-sentiment-api",
                "model_loaded": True,
                "shap_available": False
            }]
        }
    )

# ============================================================================
# Error response model (for consistent error formatting)
# ============================================================================


class ErrorResponse(BaseAPIModel):
    """Standardized error response model."""

    error: str = Field(..., description="Error type identifier")
    detail: str = Field(..., description="Human-readable description")
    # Consistent UTC timestamp for error tracking
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error UTC timestamp"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "error": "model_not_found",
                "detail": (
                    "Model bundle not found at "
                    "artifacts/model_bundle.joblib"
                ),
            }]
        }
    )
