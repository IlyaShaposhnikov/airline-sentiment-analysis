"""
Model service layer for sentiment prediction.

Handles lazy model loading, thread-safe execution, and prediction logic.
All public methods are async-safe and handle graceful degradation.

Raises:
    FileNotFoundError: If model bundle not found at configured path
    HTTPException: If prediction fails
    during execution (500) or times out (504)
"""
import asyncio
from datetime import datetime, timezone
import threading

from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool

from src.interpretability import (
    explain_prediction,
    SHAP_AVAILABLE,
    supports_word_explanations,
)
from src.models import load_model, predict_sentiment
from src.utils.logging_config import setup_logger
from .config import get_model_path, is_model_available
from .models import Explanation, PredictionRequest, PredictionResponse

logger = setup_logger(__name__)


class ModelService:
    """
    Manages model lifecycle and provides thread-safe prediction methods.

    Uses lazy loading and thread pools to keep the async event loop
    responsive during CPU-heavy inference.

    Note on multi-worker deployments:
        When running with gunicorn/uvicorn --workers > 1, each worker process
        will load its own copy of the model. This is expected behavior.
        For memory efficiency, consider using --workers=1 with thread-based
        concurrency, or implement shared memory model loading.
    """

    # Configurable timeout for prediction operations (seconds)
    PREDICTION_TIMEOUT_SEC = 30.0

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path or str(get_model_path())
        self._model = None
        self._vectorizer = None
        self._target_mapping = None
        self._target_mapping_inv = None
        self._class_names: list[str] | None = None
        self._explanations_supported = True
        self._loaded = False
        # Lock ensures sklearn thread safety during prediction
        self._predict_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory."""
        return self._loaded

    def load(self) -> None:
        """
        Load model bundle from disk. Idempotent: safe to call multiple times.

        Raises:
            FileNotFoundError: If model path is invalid or bundle missing
        """
        if self._loaded:
            return

        if not is_model_available():
            raise FileNotFoundError(
                f"Model bundle not found at: {self._model_path}\n"
                "Please run scripts/train.py first or set MODEL_PATH env var."
            )

        logger.info(f"Loading model bundle from {self._model_path}")
        (
            self._model,
            self._vectorizer,
            self._target_mapping,
            self._target_mapping_inv
        ) = load_model(
            self._model_path
        )

        # Build ordered class names for consistent indexing across predictions
        self._class_names = [
            self._target_mapping_inv.get(idx, str(idx))
            for idx in sorted(self._model.classes_)
        ]

        # Dense sentence embeddings have no word vocabulary → no word-level
        # explanations; requests with explain=true get explanation=None
        self._explanations_supported = supports_word_explanations(
            self._vectorizer
        )

        # Heavy encoders (sentence embeddings) load lazily; warm up here so
        # the first request does not pay the model-loading cost
        warmup = getattr(self._vectorizer, "warmup", None)
        if callable(warmup):
            logger.info("Warming up vectorizer...")
            warmup()

        self._loaded = True
        logger.info(f"Model loaded: classes={self._class_names}")

    @staticmethod
    def _build_prediction_response(
        text: str,
        pred_idx: int,
        proba: list[float],
        class_names: list[str],
        explanation: Explanation | None,
        timestamp: datetime,
    ) -> PredictionResponse:
        """
        Pure function: build PredictionResponse from prediction results.
        Extracted for testability — no side effects, easy to unit test.
        """
        probabilities = {
            (class_names[i] if i < len(class_names) else str(i)): float(p)
            for i, p in enumerate(proba)
        }
        confidence = float(max(proba))

        return PredictionResponse(
            text=text,
            predicted_class=(
                class_names[pred_idx]
                if pred_idx < len(class_names)
                else str(pred_idx)
            ),
            predicted_class_idx=pred_idx,
            probabilities=probabilities,
            confidence=confidence,
            timestamp=timestamp,
            explanation=explanation,
        )

    def _predict_single_sync(
        self,
        text: str,
        explain: bool,
        use_shap: bool,
        n_explain: int,
    ) -> PredictionResponse:
        """
        Synchronous prediction logic. Must be called from thread pool.

        Args:
            text: Input text to classify
            explain: Whether to generate explanation
            use_shap: Use SHAP values if available
            n_explain: Number of top contributors (from request, no fallback)

        Returns:
            Fully validated PredictionResponse

        Raises:
            HTTPException: If prediction or explanation fails
        """
        try:
            # Lock ensures thread-safe access to sklearn model during inference
            with self._predict_lock:
                # 1. Get prediction & probabilities
                pred_idx, proba = predict_sentiment(
                    self._model, self._vectorizer, text, return_proba=True
                )
                pred_idx = int(pred_idx[0])
                proba = proba[0]

                # 2. Generate explanation if requested
                explanation: Explanation | None = None
                if explain and not self._explanations_supported:
                    logger.warning(
                        "Explanation requested but not supported for "
                        f"{type(self._vectorizer).__name__}; "
                        "returning prediction without explanation"
                    )
                elif explain:
                    # Warn if SHAP requested but unavailable →
                    # fallback to weights
                    if use_shap and not SHAP_AVAILABLE:
                        logger.warning(
                            "SHAP requested but not available — "
                            "falling back to weights "
                            f"for text '{text[:50]}...'"
                        )

                    exp_result = explain_prediction(
                        self._model,
                        self._vectorizer,
                        text,
                        class_names=self._class_names,
                        n_top_contributors=n_explain,
                        use_shap=use_shap and SHAP_AVAILABLE,
                    )

                    # Safe access: explanation result may be None or malformed
                    if exp_result and isinstance(exp_result, dict):
                        method = exp_result.get("method")
                        contributors = exp_result.get("top_contributors")
                        if method is not None and contributors is not None:
                            explanation = Explanation(
                                method=method,
                                top_contributors=contributors,
                            )
                        else:
                            logger.warning(
                                "Explanation missing required keys "
                                f"for text '{text[:50]}...'"
                            )
                    else:
                        logger.warning(
                            "Explanation returned unexpected result "
                            f"for text '{text[:50]}...'"
                        )

                # 3. Build and return response via pure function
                return self._build_prediction_response(
                    text=text,
                    pred_idx=pred_idx,
                    proba=proba,
                    class_names=self._class_names,
                    explanation=explanation,
                    timestamp=datetime.now(timezone.utc),
                )

        except HTTPException:
            # Re-raise HTTP exceptions as-is (e.g., from nested calls)
            raise
        except Exception as e:
            logger.error(
                (
                    "Prediction failed for text "
                    f"'{text[:50]}...': {type(e).__name__}: {e}"
                ),
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {type(e).__name__}",
            ) from e

    async def predict_single(
        self, request: PredictionRequest
    ) -> PredictionResponse:
        """
        Async-safe single prediction.
        Offloads CPU-bound work to thread pool to avoid blocking event loop.

        Raises:
            HTTPException: 500 on prediction error, 504 on timeout
        """
        self.load()

        start_time = datetime.now(timezone.utc)
        try:
            # run_in_threadpool prevents blocking the async event loop
            # wait_for enforces PREDICTION_TIMEOUT_SEC
            # to avoid hanging requests
            result = await asyncio.wait_for(
                run_in_threadpool(
                    self._predict_single_sync,
                    request.text,
                    request.explain,
                    request.use_shap,
                    request.n_explain,
                ),
                timeout=self.PREDICTION_TIMEOUT_SEC,
            )
            duration = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            logger.debug(f"Prediction completed in {duration:.2f}s")
            return result

        except asyncio.TimeoutError:
            logger.error(
                f"Prediction timeout ({self.PREDICTION_TIMEOUT_SEC}s) "
                f"for text '{request.text[:50]}...'"
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=(
                    "Prediction timed out "
                    f"after {self.PREDICTION_TIMEOUT_SEC}s"
                ),
            ) from None
        except HTTPException:
            # Re-raise known HTTP errors without wrapping
            raise
        except Exception as e:
            # Catch unexpected errors during async orchestration
            logger.error(
                f"Async prediction wrapper failed: {e}",
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal prediction error",
            ) from e

    def predict_single_sync(
        self,
        text: str,
        explain: bool = False,
        use_shap: bool = False,
        n_explain: int = 5,
    ) -> PredictionResponse:
        """
        Synchronous public wrapper for single prediction.
        Designed for sync contexts like Streamlit dashboards.

        Args:
            text: Input text to classify
            explain: Whether to generate explanation
            use_shap: Use SHAP values if available
            n_explain: Number of top contributors

        Returns:
            PredictionResponse with classification and optional explanation
        """
        self.load()
        return self._predict_single_sync(text, explain, use_shap, n_explain)

    async def predict_batch(
        self,
        texts: list[str],
        n_explain: int,
        explain: bool = False,
        use_shap: bool = False,
    ) -> list[PredictionResponse]:
        """
        Async-safe batch prediction.
        Runs sequentially in thread pool to ensure sklearn thread safety.

        Note: Batches are processed one-by-one under the same lock.
        For high-throughput scenarios, consider model replication or
        async queue-based processing.
        Failed items are logged and skipped; the returned list may be shorter
        than the input. For strict all-or-nothing behavior, wrap calls in
        transactional logic at the API layer.
        """
        if not texts:
            return []

        self.load()

        start_time = datetime.now(timezone.utc)

        # Wrapper for thread pool execution (sequential under lock)
        def _run_batch() -> list[PredictionResponse]:
            results = []
            for i, text in enumerate(texts):
                try:
                    result = self._predict_single_sync(
                        text, explain, use_shap, n_explain
                    )
                    results.append(result)
                except Exception as e:
                    # Log but continue processing other items in batch
                    logger.error(
                        f"Batch item {i} failed: {type(e).__name__}: {e}",
                        exc_info=False,
                    )
            return results

        try:
            # Timeout scales with batch size
            # to allow longer processing for larger batches
            results = await asyncio.wait_for(
                run_in_threadpool(_run_batch),
                timeout=self.PREDICTION_TIMEOUT_SEC * len(texts),
            )
            duration = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            logger.info(
                f"Batch prediction completed: {len(results)}/{len(texts)}"
                f" successful in {duration:.2f}s"
            )
            return results

        except asyncio.TimeoutError:
            logger.error(
                "Batch prediction timeout after "
                f"{self.PREDICTION_TIMEOUT_SEC * len(texts)}s"
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Batch prediction timed out",
            ) from None
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch prediction failed",
            ) from e


# ============================================================================
# Global singleton instance for FastAPI dependency injection
# ============================================================================

# Singleton pattern: one instance shared across all FastAPI routes
model_service = ModelService()
