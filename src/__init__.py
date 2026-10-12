"""
Airline sentiment analysis package.

Windows note: torch preload (mitigation, not a proven fix)
----------------------------------------------------------
On Windows, multi-threaded sentence-embedding encoding inside
``scripts/train.py`` occasionally crashed the process with an access
violation in a torch matmul (torch 2.14 CPU, transformers 5.x).

What was measured on the author's machine:
- before this preload: 2 crashes in 2 runs;
- with it: about 1 crash in 15 runs;
- a standalone encoder script never crashed (12/12), in every import order
  of torch and scikit-learn — so the original hypothesis (scikit-learn's
  and torch's OpenMP runtimes conflicting by load order) is NOT confirmed.

Root cause unknown; the crash is rare and non-deterministic. Preloading
torch before the rest of the stack is kept because it correlated with
fewer crashes and costs nothing. If a run still crashes: rerun it, or set
``OMP_NUM_THREADS=1`` (reliable, ~2x slower encoding).

Only active on Windows and only when torch is installed; entry points
import ``src`` before scikit-learn (guarded by tests/test_runtime.py).
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
