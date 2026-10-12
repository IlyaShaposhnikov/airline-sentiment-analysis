"""
shap is imported lazily: it pulls in numba/llvmlite (slow import, heavy
native libraries), and only SHAP explanations need it.
"""
from pathlib import Path
import subprocess
import sys

import pytest

import src.interpretability as interp


def test_importing_interpretability_does_not_import_shap(
    project_root: Path,
):
    """Fresh interpreter: the module import must not load shap."""
    code = (
        "import sys; import src.interpretability; "
        "print('SHAP_LOADED=' + str('shap' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "SHAP_LOADED=False" in result.stdout


def test_get_shap_returns_none_when_import_fails(monkeypatch):
    """A broken shap install degrades to weight explanations."""
    monkeypatch.setattr(interp, "_shap_module", None)
    monkeypatch.setitem(sys.modules, "shap", None)  # import raises
    assert interp._get_shap() is None


def test_shap_installed_handles_blocked_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", None)
    assert interp._shap_installed() is False


def test_get_shap_imports_once(monkeypatch):
    shap = pytest.importorskip("shap")
    monkeypatch.setattr(interp, "_shap_module", None)
    assert interp._get_shap() is shap
    assert interp._get_shap() is shap
