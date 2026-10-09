"""
Airline sentiment analysis package.

Windows OpenMP note
-------------------
scikit-learn ships Microsoft's OpenMP runtime (``vcomp140.dll``), PyTorch
ships Intel's (``libiomp5md.dll``). If scikit-learn's runtime is loaded
first, multi-threaded torch inference (sentence embeddings) can crash the
process with an access violation. Loading torch first avoids the conflict
and keeps torch multi-threaded (``OMP_NUM_THREADS=1`` also avoids it, but
makes encoding several times slower).

The package import is the earliest common point of every entry point
(``scripts/*``, ``uvicorn src.api.main:app``, the dashboard, pytest via
conftest), so torch is preloaded here — only on Windows and only when it
is installed. Entry points must import ``src`` before scikit-learn.
"""
import importlib
import importlib.util
import sys


def _preload_torch_on_windows(
    platform: str = sys.platform,
    modules: dict = sys.modules,
    find_spec=importlib.util.find_spec,
    import_module=importlib.import_module,
) -> bool:
    """
    Import torch before scikit-learn on Windows, if torch is installed.

    Arguments are injectable for testing only.

    Returns:
        True if torch was preloaded by this call
    """
    if platform != "win32" or "torch" in modules:
        return False
    if "sklearn" in modules:
        # Too late to fix load order: the entry point imported sklearn first
        return False
    if find_spec("torch") is None:
        return False  # optional DL dependencies not installed
    import_module("torch")
    return True


TORCH_PRELOADED = _preload_torch_on_windows()
