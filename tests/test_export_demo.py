"""
scripts/export_demo.py: copies trained runs into demo/models/ for the
hosted demo, with the metadata the dashboard needs for its labels.
"""
import importlib.util
from pathlib import Path

import pytest

from src.model_registry import discover_bundles

RUN_FILES = ("model_bundle.joblib", "config_used.yaml", "metrics.json")


def _load_script(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "export_demo", project_root / "scripts" / "export_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_run(run_dir: Path, marker: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in RUN_FILES:
        (run_dir / filename).write_text(f"{marker}/{filename}",
                                        encoding="utf-8")


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    _make_run(root, "top")
    _make_run(root / "binary", "binary")
    _make_run(root / "embedding_lr", "emb")
    return root


def test_copies_and_renames_runs(project_root, artifacts, tmp_path):
    script = _load_script(project_root)
    out = tmp_path / "demo" / "models"

    bundles = script.export_runs(
        [".:tfidf_lr", "binary:tfidf_lr_binary", "embedding_lr"],
        artifacts, out,
    )

    assert sorted(p.parent.name for p in bundles) == [
        "embedding_lr", "tfidf_lr", "tfidf_lr_binary",
    ]
    for filename in RUN_FILES:
        assert (out / "tfidf_lr" / filename).read_text(
            encoding="utf-8"
        ) == f"top/{filename}"
    # The dashboard finds them as a models directory
    assert [b.name for b in discover_bundles(out)] == [
        "embedding_lr", "tfidf_lr", "tfidf_lr_binary",
    ]


def test_replaces_previous_export(project_root, artifacts, tmp_path):
    script = _load_script(project_root)
    out = tmp_path / "models"
    _make_run(out / "stale_model", "old")

    script.export_runs(["binary"], artifacts, out)

    assert [p.name for p in out.iterdir()] == ["binary"]


def test_missing_files_leave_existing_export_untouched(
    project_root, artifacts, tmp_path
):
    script = _load_script(project_root)
    out = tmp_path / "models"
    _make_run(out / "kept", "old")
    (artifacts / "binary" / "metrics.json").unlink()

    with pytest.raises(FileNotFoundError, match="metrics.json"):
        script.export_runs(["embedding_lr", "binary"], artifacts, out)

    assert [p.name for p in out.iterdir()] == ["kept"]


@pytest.mark.parametrize("runs, error", [
    (["binary:x", "embedding_lr:x"], "Duplicate"),
    (["."], "demo name"),
])
def test_invalid_run_specs(project_root, artifacts, tmp_path, runs, error):
    script = _load_script(project_root)
    with pytest.raises(ValueError, match=error):
        script.export_runs(runs, artifacts, tmp_path / "models")


def test_default_selection_matches_readme(project_root):
    """Six demo models: the 3-class grid plus two binary LogReg runs."""
    script = _load_script(project_root)
    names = [script.parse_run(spec)[1] for spec in script.DEFAULT_RUNS]
    assert names == [
        "tfidf_lr", "tfidf_mlp", "embedding_lr", "embedding_mlp",
        "tfidf_lr_binary", "embedding_lr_binary",
    ]
