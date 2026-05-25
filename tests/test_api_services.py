"""
Unit tests for src/api/services.py ModelService.

Focus: Lazy loading, thread safety, async timeout handling,
error propagation, and batch processing with partial success.

Note: These tests use mocks to isolate the service layer from
actual model loading and prediction logic.
"""

import logging
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from asyncio import TimeoutError as AsyncTimeoutError

import pytest
from fastapi import HTTPException, status

from src.api.services import ModelService, model_service
from src.api.models import PredictionRequest, PredictionResponse, Explanation


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_model_bundle():
    """Mock a loaded model bundle with minimal required attributes."""
    mock_model = MagicMock()
    mock_model.classes_ = [0, 1, 2]  # negative, positive, neutral
    mock_vectorizer = MagicMock()
    mock_target_mapping = {"negative": 0, "positive": 1, "neutral": 2}
    mock_target_mapping_inv = {0: "negative", 1: "positive", 2: "neutral"}
    return {
        "model": mock_model,
        "vectorizer": mock_vectorizer,
        "target_mapping": mock_target_mapping,
        "target_mapping_inv": mock_target_mapping_inv,
    }


@pytest.fixture
def mock_load_model(mock_model_bundle):
    """Mock the load_model function to return a fake bundle."""
    with patch("src.api.services.load_model") as mock_load:
        mock_load.return_value = (
            mock_model_bundle["model"],
            mock_model_bundle["vectorizer"],
            mock_model_bundle["target_mapping"],
            mock_model_bundle["target_mapping_inv"],
        )
        yield mock_load


@pytest.fixture
def mock_predict_sentiment():
    """Mock predict_sentiment to return controlled predictions."""
    with patch("src.api.services.predict_sentiment") as mock_pred:
        # Default: predict positive with high confidence
        mock_pred.return_value = (
            [1],  # pred_idx as array
            [[0.1, 0.8, 0.1]],  # proba as 2D array
        )
        yield mock_pred


@pytest.fixture
def mock_explain_prediction():
    """Mock explain_prediction to return controlled explanations."""
    with patch("src.api.services.explain_prediction") as mock_exp:
        mock_exp.return_value = {
            "method": "weights",
            "top_contributors": [("great", 1.5), ("flight", 0.8)],
        }
        yield mock_exp


@pytest.fixture
def mock_is_model_available():
    """Mock is_model_available to control model existence checks."""
    with patch("src.api.services.is_model_available") as mock_avail:
        mock_avail.return_value = True
        yield mock_avail


@pytest.fixture
def service_with_mock_model(mock_load_model, mock_is_model_available):
    """Create a ModelService instance with mocked model loading."""
    service = ModelService(model_path="/fake/path/model.joblib")
    service.load()  # Will use mocks
    return service


# ============================================================================
# ModelService initialization and loading tests
# ============================================================================


class TestModelServiceInitialization:
    """Tests for ModelService __init__ and properties."""

    def test_init_with_default_path(self, mock_is_model_available):
        """Should use configured default path if none provided."""
        service = ModelService()
        assert service._model_path is not None
        assert isinstance(service._model_path, str)
        assert not service._loaded  # Not loaded yet

    def test_init_with_custom_path(self):
        """Should accept custom model path."""
        custom_path = "/custom/path/model.joblib"
        service = ModelService(model_path=custom_path)
        assert service._model_path == custom_path

    def test_is_loaded_property(self, service_with_mock_model):
        """is_loaded should reflect actual load state."""
        assert service_with_mock_model.is_loaded is True

        # Create unloaded service
        service = ModelService(model_path="/fake")
        assert service.is_loaded is False


class TestModelServiceLoad:
    """Tests for the load() method."""

    def test_load_idempotent(self, mock_load_model, mock_is_model_available):
        """load() should be safe to call multiple times."""
        service = ModelService(model_path="/fake")

        # First call
        service.load()
        assert service._loaded is True
        assert mock_load_model.call_count == 1

        # Second call should not reload
        service.load()
        assert mock_load_model.call_count == 1  # Still 1

    def test_load_raises_file_not_found(self, mock_is_model_available):
        """Should raise FileNotFoundError if model not available."""
        mock_is_model_available.return_value = False

        service = ModelService(model_path="/nonexistent")

        with pytest.raises(FileNotFoundError) as exc_info:
            service.load()

        assert "Model bundle not found" in str(exc_info.value)
        assert "/nonexistent" in str(exc_info.value)

    def test_load_builds_class_names(
            self, mock_load_model, mock_is_model_available
    ):
        """Should build ordered class_names from target_mapping_inv."""
        service = ModelService(model_path="/fake")
        service.load()

        # Should be sorted by class index:
        # [0, 1, 2] → ["negative", "positive", "neutral"]
        assert service._class_names == ["negative", "positive", "neutral"]


# ============================================================================
# _build_prediction_response tests (pure function)
# ============================================================================


class TestBuildPredictionResponse:
    """Tests for the pure _build_prediction_response function."""

    def test_builds_valid_response(self):
        """Should create valid PredictionResponse from components."""
        response = ModelService._build_prediction_response(
            text="test",
            pred_idx=1,
            proba=[0.1, 0.8, 0.1],
            class_names=["negative", "positive", "neutral"],
            explanation=None,
            timestamp=datetime.now(timezone.utc),
        )

        assert response.text == "test"
        assert response.predicted_class == "positive"
        assert response.predicted_class_idx == 1
        assert response.confidence == 0.8
        assert response.probabilities["positive"] == 0.8
        assert response.explanation is None

    def test_handles_missing_class_name(self):
        """Should fallback to str(idx) if class_names is short."""
        proba = [0.5, 0.5]
        class_names = ["neg", "pos"]

        response = ModelService._build_prediction_response(
            text="test",
            pred_idx=5,
            proba=proba,
            class_names=class_names,
            explanation=None,
            timestamp=datetime.now(timezone.utc),
        )

        assert response.predicted_class == "5"
        assert response.probabilities == {"neg": 0.5, "pos": 0.5}
        assert response.confidence == 0.5

    def test_explanation_included_when_provided(self):
        """Should include explanation in response when provided."""
        explanation = Explanation(
            method="weights",
            top_contributors=[("word", 1.0)],
        )

        response = ModelService._build_prediction_response(
            text="test",
            pred_idx=0,
            proba=[1.0],
            class_names=["test"],
            explanation=explanation,
            timestamp=datetime.now(timezone.utc),
        )

        assert response.explanation is not None
        assert response.explanation.method == "weights"


# ============================================================================
# _predict_single_sync tests (core prediction logic)
# ============================================================================


class TestPredictSingleSync:
    """Tests for the synchronous prediction logic."""

    def test_basic_prediction(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
    ):
        """Should return PredictionResponse for valid input."""
        response = service_with_mock_model._predict_single_sync(
            text="Great flight!",
            explain=False,
            use_shap=False,
            n_explain=5,
        )

        assert response.predicted_class == "positive"
        assert response.confidence == 0.8
        assert response.explanation is None  # explain=False
        mock_predict_sentiment.assert_called_once()

    def test_explanation_requested(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
        mock_explain_prediction,
    ):
        """Should include explanation when explain=True."""
        response = service_with_mock_model._predict_single_sync(
            text="Great flight!",
            explain=True,
            use_shap=False,
            n_explain=5,
        )

        assert response.explanation is not None
        assert response.explanation.method == "weights"
        assert len(response.explanation.top_contributors) == 2
        mock_explain_prediction.assert_called_once()

    def test_shap_fallback_handles_weights_result(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
        mock_explain_prediction,
    ):
        """
        Should correctly process explanation when use_shap=True.
        Tests fallback behavior by mocking explain_prediction
        to return weights-based result.
        """
        mock_explain_prediction.return_value = {
            "method": "weights",
            "top_contributors": [("flight", 1.2), ("great", 0.8)],
        }

        response = service_with_mock_model._predict_single_sync(
            text="test",
            explain=True,
            use_shap=True,
            n_explain=5,
        )

        assert response.explanation is not None
        assert response.explanation.method == "weights"
        assert len(response.explanation.top_contributors) == 2

        mock_explain_prediction.assert_called_once()

    def test_prediction_error_wrapped_in_http_exception(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
    ):
        """Should wrap unexpected errors in HTTPException(500)."""
        mock_predict_sentiment.side_effect = RuntimeError("Model crashed")

        with pytest.raises(HTTPException) as exc_info:
            service_with_mock_model._predict_single_sync(
                text="test",
                explain=False,
                use_shap=False,
                n_explain=5,
            )

        assert (
            exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        assert "Prediction failed: RuntimeError" in exc_info.value.detail

    def test_http_exception_propagated_unchanged(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
    ):
        """Should re-raise HTTPException without wrapping."""
        original_error = HTTPException(
            status_code=400, detail="Bad request from nested call"
        )
        mock_predict_sentiment.side_effect = original_error

        with pytest.raises(HTTPException) as exc_info:
            service_with_mock_model._predict_single_sync(
                text="test",
                explain=False,
                use_shap=False,
                n_explain=5,
            )

        # Should be the same exception, not wrapped
        assert exc_info.value is original_error
        assert exc_info.value.status_code == 400

    def test_empty_text_handling(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
    ):
        """Empty text should be handled gracefully by model/vectorizer."""
        response = service_with_mock_model._predict_single_sync(
            text="", explain=False, use_shap=False, n_explain=5
        )
        assert isinstance(response, PredictionResponse)

    def test_explanation_none_result(
        self,
        mock_load_model,
        mock_is_model_available,
        mock_predict_sentiment,
        caplog,
    ):
        """
        If explain_prediction returns None,
        explanation should be None in response.
        """
        service = ModelService(model_path="/fake")
        service.load()

        with patch("src.api.services.explain_prediction") as mock_exp:
            mock_exp.return_value = None

            with caplog.at_level(logging.WARNING, logger="src.api.services"):
                response = service._predict_single_sync(
                    text="test", explain=True, use_shap=False, n_explain=5
                )

            assert response.explanation is None
            assert any(
                "Explanation returned unexpected result" in r.message
                for r in caplog.records
            ), (
                "Expected warning for None explanation, "
                f"got: {[r.message for r in caplog.records]}"
            )

    def test_class_names_with_missing_mapping(
        self,
        mock_load_model,
        mock_is_model_available,
    ):
        """Should handle missing keys in target_mapping_inv gracefully."""
        mock_model = MagicMock()
        mock_model.classes_ = [0, 1, 99]
        mock_vectorizer = MagicMock()
        mock_target_mapping = {"negative": 0, "positive": 1}
        mock_target_mapping_inv = {0: "negative", 1: "positive"}

        mock_load_model.return_value = (
            mock_model, mock_vectorizer,
            mock_target_mapping, mock_target_mapping_inv
        )

        service = ModelService(model_path="/fake")
        service.load()

        assert "99" in service._class_names
        assert service._class_names == ["negative", "positive", "99"]


# ============================================================================
# predict_single tests (async wrapper with timeout)
# ============================================================================


class TestPredictSingleAsync:
    """Tests for the async predict_single method."""

    @pytest.mark.asyncio
    async def test_successful_async_prediction(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
    ):
        """Should return PredictionResponse in async context."""
        request = PredictionRequest(text="Great flight!")

        response = await service_with_mock_model.predict_single(request)

        assert response.predicted_class == "positive"
        mock_predict_sentiment.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_timeout_raises_504(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
    ):
        """Should raise HTTPException(504) when asyncio.wait_for times out."""
        with patch("src.api.services.asyncio.wait_for") as mock_wait:
            mock_wait.side_effect = AsyncTimeoutError()

            request = PredictionRequest(text="test")

            with pytest.raises(HTTPException) as exc_info:
                await service_with_mock_model.predict_single(request)

            assert (
                exc_info.value.status_code == status.HTTP_504_GATEWAY_TIMEOUT
            )
            assert "timed out" in exc_info.value.detail.lower()
            mock_wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_error_wrapped_in_500(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
    ):
        """Should wrap unexpected async errors in HTTPException(500)."""
        mock_predict_sentiment.side_effect = ValueError("Unexpected error")

        request = PredictionRequest(text="test")

        with pytest.raises(HTTPException) as exc_info:
            await service_with_mock_model.predict_single(request)

        assert (
            exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ============================================================================
# predict_single_sync public wrapper tests
# ============================================================================


class TestPredictSingleSyncPublic:
    """Tests for the public sync wrapper predict_single_sync()."""

    def test_delegates_to_private_method(
        self,
        mock_load_model,
        mock_is_model_available,
        mock_predict_sentiment,
        mock_explain_prediction,
    ):
        """Should call _predict_single_sync with same args."""
        service = ModelService(model_path="/fake")
        service.load()

        # Spy on private method
        with patch.object(
            service,
            "_predict_single_sync",
            wraps=service._predict_single_sync
        ) as spy:

            response = service.predict_single_sync(
                text="test",
                explain=True,
                use_shap=False,
                n_explain=10,
            )

            # Should have called private method with exact args
            spy.assert_called_once_with("test", True, False, 10)
            assert response is not None

    def test_ensures_model_loaded(
        self,
        mock_load_model,
        mock_is_model_available,
        mock_predict_sentiment,
    ):
        """Should call load() before prediction if not loaded."""
        service = ModelService(model_path="/fake")
        assert not service._loaded

        # Mock the private method to avoid actual prediction
        with patch.object(service, "_predict_single_sync") as mock_private:
            mock_private.return_value = MagicMock()

            service.predict_single_sync(text="test")

            # Should have loaded model first
            assert service._loaded is True
            mock_load_model.assert_called_once()


# ============================================================================
# predict_batch tests (batch processing with partial success)
# ============================================================================


class TestPredictBatch:
    """Tests for the async batch prediction method."""

    @pytest.mark.asyncio
    async def test_successful_batch(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
    ):
        """Should return list of PredictionResponse for all texts."""
        texts = ["text1", "text2", "text3"]

        results = await service_with_mock_model.predict_batch(
            texts=texts,
            n_explain=5,
            explain=False,
            use_shap=False,
        )

        assert len(results) == 3
        assert all(r.predicted_class == "positive" for r in results)
        # Should call predict_sentiment once per text
        assert mock_predict_sentiment.call_count == 3

    @pytest.mark.asyncio
    async def test_partial_success_continues_on_error(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
        caplog,
    ):
        """Should continue processing batch after individual failures."""
        # Mock: succeed on first and third, fail on second
        def side_effect(*args, **kwargs):
            text = args[2] if len(args) > 2 else ""
            if "fail" in text.lower():
                raise RuntimeError("Simulated failure")
            return ([1], [[0.1, 0.8, 0.1]])

        mock_predict_sentiment.side_effect = side_effect

        texts = ["good1", "will_fail", "good2"]

        results = await service_with_mock_model.predict_batch(
            texts=texts,
            n_explain=5,
            explain=False,
            use_shap=False,
        )

        # Should have 2 successful results (failed items are skipped)
        assert len(results) == 2
        # Should log the failure
        assert any("Batch item 1 failed" in r.message for r in caplog.records)
        assert any("Simulated failure" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_batch_timeout(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
    ):
        """
        Should raise HTTPException(504) when batch asyncio.wait_for times out.
        """
        with patch("src.api.services.asyncio.wait_for") as mock_wait:
            mock_wait.side_effect = AsyncTimeoutError()

            texts = ["t1", "t2", "t3"]

            with pytest.raises(HTTPException) as exc_info:
                await service_with_mock_model.predict_batch(
                    texts=texts,
                    n_explain=5,
                    explain=False,
                    use_shap=False,
                )

            assert (
                exc_info.value.status_code == status.HTTP_504_GATEWAY_TIMEOUT
            )
            assert "Batch prediction timed out" in exc_info.value.detail
            mock_wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_empty_list(self, service_with_mock_model):
        """Empty batch should return empty list without errors."""
        results = await service_with_mock_model.predict_batch(
            texts=[], n_explain=5, explain=False, use_shap=False
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_all_failures(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
        caplog,
    ):
        """If all batch items fail, should return empty list and log errors."""
        mock_predict_sentiment.side_effect = RuntimeError("Always fail")

        with caplog.at_level(logging.ERROR, logger="src.api.services"):
            results = await service_with_mock_model.predict_batch(
                texts=["a", "b"], n_explain=5, explain=False, use_shap=False
            )

        assert results == []
        assert caplog.records, "Expected error logs for failed batch items"
        assert any("Batch item" in r.message for r in caplog.records)


# ============================================================================
# Thread safety tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe execution with _predict_lock."""

    def test_lock_acquired_during_prediction(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
    ):
        """Should have _predict_lock attribute of correct type."""
        assert hasattr(service_with_mock_model, "_predict_lock")
        assert isinstance(
            service_with_mock_model._predict_lock,
            type(threading.Lock())
        )

        response = service_with_mock_model._predict_single_sync(
            text="test",
            explain=False,
            use_shap=False,
            n_explain=5,
        )
        assert response is not None

    def test_concurrent_calls_serialize(
        self,
        service_with_mock_model,
        mock_predict_sentiment,
    ):
        """
        Multiple threads should execute predictions sequentially under lock.
        """
        enter_event = threading.Event()
        exit_event = threading.Event()
        execution_log = []

        def controlled_predict(*args, **kwargs):
            thread_id = threading.current_thread().ident
            execution_log.append(f"enter-{thread_id}")

            enter_event.set()
            exit_event.wait(timeout=5)

            execution_log.append(f"exit-{thread_id}")
            return ([1], [[0.1, 0.8, 0.1]])

        mock_predict_sentiment.side_effect = controlled_predict

        def run_prediction(text):
            service_with_mock_model._predict_single_sync(
                text, False, False, 5
            )

        t1 = threading.Thread(
            target=run_prediction, args=("text1",), name="T1"
        )
        t2 = threading.Thread(
            target=run_prediction, args=("text2",), name="T2"
        )

        t1.start()
        t2.start()
        assert enter_event.wait(timeout=2), "Thread 1 didn't acquire lock"

        assert len([e for e in execution_log if "enter" in e]) == 1, \
            "Thread 2 entered before Thread 1 finished (lock not working)"

        exit_event.set()
        t1.join(timeout=2)
        t2.join(timeout=2)

        assert len(execution_log) == 4, (
            "Expected 4 log entries, "
            f"got {len(execution_log)}: {execution_log}"
        )

        first_enter = execution_log[0]
        thread_id = first_enter.split("-")[1]

        enter_idx = next(
            i for i, e in enumerate(execution_log)
            if e == f"enter-{thread_id}"
        )
        exit_idx = next(
            i for i, e in enumerate(execution_log)
            if e == f"exit-{thread_id}"
        )

        assert exit_idx > enter_idx, f"Lock order violated: {execution_log}"

        between = execution_log[enter_idx+1:exit_idx]
        assert all("enter" not in e for e in between), \
            f"Another thread entered during critical section: {execution_log}"


# ============================================================================
# Global singleton tests
# ============================================================================


class TestGlobalSingleton:
    """Tests for the global model_service instance."""

    def test_singleton_exists(self):
        """Global model_service should be a ModelService instance."""
        assert isinstance(model_service, ModelService)

    def test_singleton_not_preloaded(self):
        """Singleton should not be pre-loaded at import time."""
        # The global instance is created but not loaded until first use
        assert model_service.is_loaded is False
