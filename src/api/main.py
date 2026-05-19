"""
FastAPI application entry point for Airline Sentiment Analysis API.

Provides REST endpoints with typed request/response validation,
async-safe prediction execution, and comprehensive error handling.

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /              # API info
    GET  /health        # Health check
    POST /predict       # Single text prediction
    POST /predict/batch # Batch prediction (1-100 texts)
    GET  /docs          # Swagger UI (interactive docs)
    GET  /redoc         # ReDoc (alternative docs)
"""

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    API_DOCS_URL,
    API_REDOC_URL,
    API_ENABLED,
    API_HOST,
    API_PORT,
    CORS_ALLOWED_ORIGINS,
)
from .models import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse,
)
from .services import model_service, SHAP_AVAILABLE
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Custom exception handlers (named functions for proper logging)
# ============================================================================

def _handle_404(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle 404 errors with logging."""
    logger.warning(f"404 Not Found: {request.method} {request.url}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse(
            error="not_found",
            detail="Endpoint not found",
        ).model_dump(mode='json'),
    )


def _handle_422(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle 422 validation errors with logging."""
    errors_summary = [
        f"{err['loc']}: {err['msg']}" for err in exc.errors()[:3]
    ]  # Limit to first 3 for brevity
    logger.warning(
        f"422 Validation Error on {request.url}: {'; '.join(errors_summary)}"
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="validation_error",
            detail="Request validation failed",
        ).model_dump(mode='json'),
    )


# ============================================================================
# FastAPI application instance
# ============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url=API_DOCS_URL,
    redoc_url=API_REDOC_URL,
    exception_handlers={
        404: _handle_404,
        422: _handle_422,
    },
    servers=[{
        "url": "/",
        "description": "Current host"
    }],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Lifecycle events
# ============================================================================


@app.on_event("startup")
async def startup_event() -> None:
    """
    Pre-load model on application startup if API is enabled.

    This ensures the first request doesn't pay the model loading penalty.
    """
    if not API_ENABLED:
        logger.warning("API is disabled in config — skipping model preload")
        return

    try:
        model_service.load()
        # Smoke test: verify model can actually predict
        _ = model_service._build_prediction_response(
            text="smoke_test",
            pred_idx=0,
            proba=[1.0],
            class_names=["test"],
            explanation=None,
            timestamp=datetime.now(timezone.utc),
        )
        logger.info("Model pre-loaded and smoke-tested successfully")
    except FileNotFoundError as e:
        logger.warning(f"Model not found at startup: {e}")
        logger.warning("Model will be loaded on first prediction request")
    except Exception as e:
        logger.error(f"Failed to preload or test model: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Cleanup on application shutdown.

    Currently no cleanup needed, but reserved for future use
    (e.g., closing database connections, flushing logs).
    """
    logger.info("API shutdown initiated")


# ============================================================================
# API endpoints
# ============================================================================


@app.get("/", tags=["Root"], response_model=dict)
async def root() -> dict:
    """
    Root endpoint with API information and available endpoints.

    Returns:
        Dictionary with service metadata and endpoint documentation links.
    """
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "documentation": {
            "swagger_ui": API_DOCS_URL,
            "redoc": API_REDOC_URL,
        },
        "endpoints": {
            "GET /": "This endpoint — API information",
            "GET /health": "Health check with model status",
            "POST /predict": "Single text sentiment prediction",
            "POST /predict/batch": "Batch prediction (1-100 texts)",
        },
        "config": {
            "host": API_HOST,
            "port": API_PORT,
            "enabled": API_ENABLED,
        },
    }


@app.get(
    "/health",
    tags=["Health"],
    response_model=HealthResponse,
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service unavailable", "model": ErrorResponse},
    },
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for load balancers and monitoring.

    Returns:
        HealthResponse with service status, model load state,
        and SHAP availability.

    Raises:
        HTTPException:
            - 503: API is disabled in configuration
    """
    if not API_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API is disabled in configuration",
        )

    model_operational = model_service.is_loaded
    if model_operational:
        try:
            # Smoke test: valid response
            _ = model_service._build_prediction_response(
                text="health_check",
                pred_idx=0,
                proba=[1.0],
                class_names=["test"],
                explanation=None,
                timestamp=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.error(f"Health check: model operational test failed: {e}")
            model_operational = False

    return HealthResponse(
        timestamp=datetime.now(timezone.utc),
        model_loaded=model_operational,
        shap_available=SHAP_AVAILABLE,
    )


@app.post(
    "/predict",
    tags=["Prediction"],
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid request", "model": ErrorResponse},
        500: {"description": "Prediction failed", "model": ErrorResponse},
        504: {"description": "Prediction timed out", "model": ErrorResponse},
    },
)
async def predict_single(request: PredictionRequest) -> PredictionResponse:
    """
    Predict sentiment for a single text.

    Args:
        request: PredictionRequest with text and optional explanation settings

    Returns:
        PredictionResponse with classification, probabilities,
        and optional explanation.

    Raises:
        HTTPException:
            - 400: Request validation failed (Pydantic)
            - 500: Prediction execution error or internal failure
            - 504: Prediction timed out (configurable timeout)
    """
    try:
        result = await model_service.predict_single(request)

        truncated_text = (
            f"{request.text[:50]}"
            f"{'...' if len(request.text) > 50 else ''}"
        )
        logger.info(
            f"Prediction: '{truncated_text}' "
            f"→ {result.predicted_class} (confidence: {result.confidence:.2f})"
        )

        return result

    except HTTPException:
        # Re-raise known HTTP errors (e.g., timeout from service layer)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction",
        ) from e


@app.post(
    "/predict/batch",
    tags=["Prediction"],
    response_model=BatchPredictionResponse,
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"description": "Invalid request", "model": ErrorResponse},
        500: {
            "description": "Batch prediction failed", "model": ErrorResponse
        },
        504: {
            "description": "Batch prediction timed out", "model": ErrorResponse
        },
    },
)
async def predict_batch(
    request: BatchPredictionRequest
) -> BatchPredictionResponse:
    """
    Predict sentiment for multiple texts in a single request.

    Args:
        request: BatchPredictionRequest with list of texts
        and optional explanation settings

    Returns:
        BatchPredictionResponse with list of predictions and processing time.

    Raises:
        HTTPException:
            - 400: Request validation failed (Pydantic)
            - 500: Batch prediction execution error or internal failure
            - 504: Batch prediction timed out (scalable timeout)
    """
    start = datetime.now(timezone.utc)

    try:
        predictions = await model_service.predict_batch(
            texts=request.texts,
            explain=request.explain,
            use_shap=request.use_shap,
            n_explain=request.n_explain,
        )

        duration_ms = (
            (datetime.now(timezone.utc) - start).total_seconds() * 1000
        )

        logger.info(
            f"Batch prediction: {len(predictions)}"
            f"/{len(request.texts)} successful "
            f"in {duration_ms:.1f}ms"
        )

        return BatchPredictionResponse(
            count=len(predictions),
            predictions=predictions,
            processing_time_ms=max(0.01, round(duration_ms, 2)),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /predict/batch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during batch prediction",
        ) from e


# ============================================================================
# Global exception handler for unhandled errors
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all exception handler for consistent error responses.

    Logs the error and returns a standardized ErrorResponse JSON.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url}: {exc}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            detail="An unexpected error occurred",
        ).model_dump(mode='json'),
    )
