"""
Discovery of trained model bundles (used by the dashboard).

``scripts/train.py`` writes ``metrics.json`` and ``config_used.yaml`` next to
every ``model_bundle.joblib``. Listing models reads only those small files,
so it is instant and loads neither the bundles nor torch.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path

import yaml

BUNDLE_FILENAME = "model_bundle.joblib"
DEFAULT_BUNDLE_NAME = "default"

# Where the dashboard looks for models (relative to the project root):
# local training runs first, then the small committed demo set used by the
# hosted demo (Streamlit Community Cloud has no artifacts/)
ARTIFACTS_DIRNAME = "artifacts"
DEMO_MODELS_DIRNAME = "demo/models"
MODELS_DIR_ENV = "DASHBOARD_MODELS_DIR"

# Preferred initial selection when MODEL_PATH is not among the bundles
FALLBACK_NAMES = (DEFAULT_BUNDLE_NAME, "tfidf_lr")

VECTORIZER_LABELS = {
    "tfidf": "TF-IDF",
    "count": "Count",
    "sentence_embedding": "Embeddings",
}
MODEL_LABELS = {
    "logistic_regression": "LogReg",
    "mlp": "MLP",
}


@dataclass(frozen=True)
class BundleInfo:
    """What the dashboard shows about a bundle before loading it."""

    path: Path
    name: str
    model_type: str | None = None
    vectorizer_type: str | None = None
    task_type: str | None = None
    accuracy: float | None = None
    f1_macro: float | None = None

    @property
    def description(self) -> str | None:
        """E.g. "TF-IDF + LogReg"; None if the training config is missing."""
        if not (self.model_type and self.vectorizer_type):
            return None
        vec = VECTORIZER_LABELS.get(
            self.vectorizer_type, self.vectorizer_type
        )
        model = MODEL_LABELS.get(self.model_type, self.model_type)
        return f"{vec} + {model}"

    @property
    def label(self) -> str:
        """One-line label, e.g. "embedding_lr: Embeddings + LogReg ·
        3 classes · F1 0.805"."""
        parts = [self.description or "details unknown"]
        if self.task_type == "binary":
            parts.append("binary")
        elif self.task_type == "multiclass":
            parts.append("3 classes")
        if self.f1_macro is not None:
            parts.append(f"F1 {self.f1_macro:.3f}")
        return f"{self.name}: " + " · ".join(parts)

    @property
    def needs_deep_learning(self) -> bool:
        """True if loading requires torch / sentence-transformers."""
        return (
            self.model_type == "mlp"
            or self.vectorizer_type == "sentence_embedding"
        )


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_yaml(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def describe_bundle(bundle_path: Path, artifacts_dir: Path) -> BundleInfo:
    """
    Build a BundleInfo from the files next to ``bundle_path``.

    Missing or unreadable metadata is not an error: the bundle is still
    listed, just with fewer details.
    """
    bundle_path = Path(bundle_path)
    run_dir = bundle_path.parent
    try:
        relative = run_dir.relative_to(artifacts_dir)
        name = relative.as_posix() if relative.parts else DEFAULT_BUNDLE_NAME
    except ValueError:
        name = run_dir.name

    config = _read_yaml(run_dir / "config_used.yaml")
    metrics = _read_json(run_dir / "metrics.json")
    model_cfg = config.get("model") or {}
    vectorizer_cfg = (config.get("preprocessing") or {}).get("vectorizer")

    return BundleInfo(
        path=bundle_path,
        name=name,
        model_type=model_cfg.get("type"),
        vectorizer_type=(vectorizer_cfg or {}).get("type"),
        task_type=metrics.get("task_type"),
        accuracy=_as_float(metrics.get("accuracy")),
        f1_macro=_as_float(metrics.get("f1_macro")),
    )


def discover_bundles(artifacts_dir: str | Path) -> list[BundleInfo]:
    """
    Find ``artifacts_dir/model_bundle.joblib`` and
    ``artifacts_dir/<run>/model_bundle.joblib``.

    Returns:
        BundleInfo list: the top-level (default) bundle first, then runs in
        alphabetical order; empty if the directory does not exist
    """
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.is_dir():
        return []
    candidates = [artifacts_dir / BUNDLE_FILENAME] + sorted(
        artifacts_dir.glob(f"*/{BUNDLE_FILENAME}")
    )
    return [
        describe_bundle(path, artifacts_dir)
        for path in candidates if path.is_file()
    ]


def default_bundle_index(
    bundles: list[BundleInfo],
    preferred: str | Path,
    fallback_names: tuple[str, ...] = FALLBACK_NAMES,
) -> int:
    """
    Index of the bundle to select first.

    ``preferred`` (e.g. MODEL_PATH) if present, else the first bundle named
    in ``fallback_names`` (the fast, explainable TF-IDF + LogReg), else 0.
    """
    try:
        preferred = Path(preferred).resolve()
    except OSError:
        preferred = None
    for i, bundle in enumerate(bundles):
        try:
            if preferred is not None and bundle.path.resolve() == preferred:
                return i
        except OSError:
            continue
    names = [b.name for b in bundles]
    for name in fallback_names:
        if name in names:
            return names.index(name)
    return 0


def choose_models_dir(
    project_root: str | Path, env: Mapping[str, str] | None = None
) -> Path:
    """
    Directory the dashboard lists models from.

    1. ``DASHBOARD_MODELS_DIR`` if set (relative to the project root);
    2. ``artifacts/`` if it contains at least one bundle (local training);
    3. ``demo/models/`` (committed demo set, used by the hosted demo).
    """
    project_root = Path(project_root)
    env = os.environ if env is None else env
    override = env.get(MODELS_DIR_ENV)
    if override:
        path = Path(override)
        return path if path.is_absolute() else project_root / path
    artifacts = project_root / ARTIFACTS_DIRNAME
    if discover_bundles(artifacts):
        return artifacts
    return project_root / DEMO_MODELS_DIRNAME
