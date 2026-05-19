"""
Pydantic v2 models for API request/response validation.

All input/output data is validated against these schemas before processing.
"""

from datetime import datetime, timezone
from typing import Annotated, Dict, List, Literal, Tuple, Optional

from pydantic import (
    BaseModel, Field, field_validator, model_validator, ConfigDict
)

from .config import (
    MAX_TEXT_LENGTH,
    MAX_BATCH_SIZE,
    MIN_EXPLAIN_COUNT,
    MAX_EXPLAIN_COUNT,
)


# ============================================================================
# Base models & Config
# ============================================================================


class BaseAPIModel(BaseModel):
    """Base config for all API models."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )


class BasePredictionRequest(BaseAPIModel):
    """Shared fields for prediction requests."""
    n_explain: Annotated[
        int,
        Field(
            default=MIN_EXPLAIN_COUNT,
            ge=MIN_EXPLAIN_COUNT,
            le=MAX_EXPLAIN_COUNT
        )
    ]


class Explanation(BaseAPIModel):
    """Typed explanation schema."""
    method: Literal["weights", "shap"] = Field(
        ..., description="Explanation method used"
    )
    top_contributors: List[Tuple[str, float]] = Field(
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
    """
    Request model for single text sentiment prediction.

    Attributes:
        text: The tweet text to analyze (1-1000 chars after stripping)
        explain: Whether to include explanation with top contributing words
        use_shap: Whether to use SHAP values instead of model weights
        n_explain: Number of top contributors to return (1-20)
    """

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
    """
    Request model for batch sentiment prediction.

    Attributes:
        texts: List of 1-100 tweet texts to analyze
        explain: Whether to include explanations for all predictions
        use_shap: Whether to use SHAP values for explanations
        n_explain: Number of top contributors per text (1-20)
    """

    texts: Annotated[
        List[str],
        Field(
            ...,
            min_length=1,
            max_length=MAX_BATCH_SIZE,
            description="List of texts to analyze for sentiment",
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
    def validate_texts(cls, v: List[str]) -> List[str]:
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
    """
    Response model for single text sentiment prediction.

    Attributes:
        text: The original input text
        predicted_class: Human-readable sentiment label
        predicted_class_idx: Numeric class index (0, 1, 2...)
        probabilities: Dict mapping class names to probability scores
        confidence: Highest probability score (model's confidence)
        timestamp: ISO format timestamp of prediction
        explanation: Optional explanation with top contributing words
    """

    text: str = Field(..., description="Original input text")
    predicted_class: str = Field(..., description="Predicted sentiment label")
    predicted_class_idx: int = Field(..., description="Numeric class index")
    probabilities: Dict[str, float] = Field(
        ..., description="Probability scores for each class"
    )
    confidence: Annotated[float, Field(
        ..., ge=0.0, le=1.0, description="Model confidence")]
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of prediction"
    )
    explanation: Optional[Explanation] = Field(
        None, description="Typed explanation if requested"
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
                "timestamp": "2026-05-18T10:00:00.123456",
                "explanation": {
                    "method": "weights",
                    "top_contributors": [["great", 1.85], ["flight", 0.45]]
                }
            }]
        }
    )


class BatchPredictionResponse(BaseAPIModel):
    """
    Response model for batch sentiment prediction.

    Attributes:
        status: Overall status of the batch operation
        count: Number of predictions returned
        predictions: List of individual PredictionResponse objects
        processing_time_ms: Total processing time in milliseconds
    """

    status: Annotated[
        str, Field("success", description="Overall status")
    ] = "success"
    count: Annotated[int, Field(
        ..., ge=0, description="Number of predictions"
    )]
    predictions: List[PredictionResponse] = Field(
        ..., description="List of results"
    )
    processing_time_ms: Annotated[float, Field(
        ..., ge=0.0, description="Processing time"
    )]

    @model_validator(mode='after')
    def validate_count_consistency(self) -> 'BatchPredictionResponse':
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
    """
    Response model for health check endpoint.

    Attributes:
        status: Health status indicator
        service: Service name identifier
        timestamp: Current timestamp
        model_loaded: Whether the model is currently loaded in memory
        shap_available: Whether SHAP library is available for explanations
    """

    status: Annotated[str, Field(
        "healthy", description="Health status"
    )] = "healthy"
    service: Annotated[str, Field(
        "airline-sentiment-api", description="Service name"
    )] = "airline-sentiment-api"
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
                "timestamp": "2026-05-18T10:00:00.123456",
                "model_loaded": True,
                "shap_available": False
            }]
        }
    )

# ============================================================================
# Error response model (for consistent error formatting)
# ============================================================================


class ErrorResponse(BaseAPIModel):
    """
    Standardized error response model.

    Attributes:
        error: Error type identifier
        detail: Human-readable error description
        timestamp: When the error occurred
    """

    error: str = Field(..., description="Error type identifier")
    detail: str = Field(..., description="Human-readable description")
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
                "timestamp": "2026-05-18T10:00:00.123456"
            }]
        }
    )
