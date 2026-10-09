"""
Tests for the Windows OpenMP workaround in src/__init__.py.

scikit-learn (vcomp140.dll) and torch (libiomp5md.dll) ship different
OpenMP runtimes; on Windows, multi-threaded torch inference crashes if
scikit-learn's runtime is loaded first. The package preloads torch, which
only works if entry points import ``src`` before ``sklearn``.
"""
import ast
from pathlib import Path

import pytest

from src import _preload_torch_on_windows


def _call(platform="win32", modules=None, torch_installed=True):
    imported = []
    result = _preload_torch_on_windows(
        platform=platform,
        modules={} if modules is None else modules,
        find_spec=lambda name: object() if torch_installed else None,
        import_module=imported.append,
    )
    return result, imported


class TestPreloadTorchOnWindows:

    def test_preloads_on_windows_when_installed(self):
        assert _call() == (True, ["torch"])

    def test_noop_on_other_platforms(self):
        assert _call(platform="linux") == (False, [])

    def test_noop_when_torch_not_installed(self):
        """Baseline TF-IDF setup without DL dependencies is untouched."""
        assert _call(torch_installed=False) == (False, [])

    def test_noop_when_torch_already_loaded(self):
        assert _call(modules={"torch": object()}) == (False, [])

    def test_noop_when_sklearn_loaded_first(self):
        """Preloading after sklearn would not fix the load order."""
        assert _call(modules={"sklearn": object()}) == (False, [])


# Packages that load scikit-learn's OpenMP runtime when imported
# (shap imports scikit-learn at import time)
_LOADS_SKLEARN = {"sklearn", "shap"}

# Files executed directly as scripts (not imported as part of ``src``):
# ``uvicorn src.api.main:app`` needs no check — importing ``src.api.main``
# runs ``src/__init__.py`` before anything else
_ENTRY_POINTS = ["scripts/train.py", "scripts/predict.py", "src/dashboard.py"]


def _module_level_imports(path: Path) -> list[str]:
    """Top-level package names imported at module level, in source order."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.append(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS)
def test_entry_point_imports_src_before_sklearn(
    entry_point, project_root: Path
):
    """
    Regression guard (static, runs on every OS): an entry point that imports
    scikit-learn (directly or via shap) before ``src`` silently disables the
    torch preload and brings back the Windows access-violation crash.
    """
    imports = _module_level_imports(project_root / entry_point)
    assert "src" in imports, f"{entry_point} does not import src"

    before_src = imports[: imports.index("src")]
    offending = sorted(_LOADS_SKLEARN.intersection(before_src))
    assert not offending, (
        f"{entry_point} imports {offending} before src; "
        "move these imports below the src imports (see src/__init__.py)"
    )
