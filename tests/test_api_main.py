"""
API wiring: the configured log level is applied, and the validation error
handler returns 422 without Starlette deprecation warnings.
"""
import json
import logging
from types import SimpleNamespace
import warnings

import pytest
from fastapi.exceptions import RequestValidationError

from src.api import main
from src.utils.logging_config import set_package_log_level, setup_logger


@pytest.fixture
def restore_src_levels():
    """Put project loggers back to INFO after a test changes them."""
    yield
    set_package_log_level("INFO")


class TestSetPackageLogLevel:
    def test_updates_existing_loggers_of_package(self):
        a = setup_logger("pkglevel_demo.a")
        b = setup_logger("pkglevel_demo.sub.b")

        applied = set_package_log_level("warning", prefix="pkglevel_demo")

        assert applied == "WARNING"
        assert a.level == logging.WARNING
        assert b.level == logging.WARNING

    def test_leaves_other_packages_alone(self):
        other = setup_logger("pkglevel_demo_other.c")
        set_package_log_level("ERROR", prefix="pkglevel_demo")
        assert other.level == logging.INFO

    def test_invalid_level_rejected(self):
        with pytest.raises(ValueError):
            set_package_log_level("LOUD", prefix="pkglevel_demo")


class TestApiLogLevel:
    def test_configured_level_reaches_service_logger(
        self, restore_src_levels
    ):
        service_logger = logging.getLogger("src.api.services")

        main._apply_log_level("ERROR")

        assert service_logger.level == logging.ERROR
        assert logging.getLogger("src.models").level == logging.ERROR

    def test_invalid_level_falls_back_to_info(self, restore_src_levels):
        assert main._apply_log_level("LOUD") == "INFO"
        assert logging.getLogger("src.api.services").level == logging.INFO

    def test_main_uses_configured_level(self):
        # main must apply the config value (serving.api.log_level or
        # API_LOG_LEVEL) at import, not a hard-coded level
        source = open(main.__file__, encoding="utf-8").read()
        assert "_apply_log_level(LOG_LEVEL)" in source


def test_validation_handler_returns_422_without_deprecation_warning():
    exc = RequestValidationError(
        [{"loc": ("body", "text"), "msg": "Field required",
          "type": "missing"}]
    )
    request = SimpleNamespace(method="POST", url="http://test/predict")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        response = main._handle_422(request, exc)

    assert response.status_code == 422
    assert json.loads(response.body)["error"] == "validation_error"
    # StarletteDeprecationWarning is not a DeprecationWarning subclass in
    # every version, so match on the message
    deprecations = [w for w in caught if "deprecated" in str(w.message)]
    assert not deprecations, [str(w.message) for w in deprecations]
