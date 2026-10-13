#!/usr/bin/env python3
"""
Copy trained runs into demo/models/ for the hosted Streamlit demo.

artifacts/ is git-ignored; the demo needs a small committed set of bundles.
Each run is copied with the metadata the dashboard reads (no retraining):

    model_bundle.joblib, config_used.yaml, metrics.json

Usage:
    python scripts/export_demo.py              # default selection below
    python scripts/export_demo.py binary:tfidf_lr_binary embedding_lr

A run is given as SOURCE[:NAME]: SOURCE is a folder under artifacts/
("." for the top-level bundle), NAME the folder name in demo/models/.
"""
import argparse
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_FILES = ("model_bundle.joblib", "config_used.yaml", "metrics.json")

# The comparison grid for 3 classes plus the two binary LogReg runs
DEFAULT_RUNS = [
    ".:tfidf_lr",
    "tfidf_mlp",
    "embedding_lr",
    "embedding_mlp",
    "binary:tfidf_lr_binary",
    "embedding_lr_binary",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy trained runs into demo/models/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "runs", nargs="*", default=DEFAULT_RUNS,
        help="SOURCE[:NAME] pairs; SOURCE is relative to --artifacts-dir",
    )
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--output-dir", default="demo/models")
    return parser.parse_args(argv)


def _resolve(path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_run(spec: str) -> tuple[str, str]:
    """"binary:tfidf_lr_binary" -> ("binary", "tfidf_lr_binary")."""
    source, _, name = spec.partition(":")
    name = name or Path(source).name
    if not name or name in (".", ".."):
        raise ValueError(f"Give a demo name for {spec!r}, e.g. {spec}:my_run")
    return source, name


def export_runs(
    runs: list[str], artifacts_dir: Path, output_dir: Path
) -> list[Path]:
    """
    Copy each run's files into ``output_dir/<name>/``.

    All runs are validated first, so a missing file leaves the existing
    demo set untouched. ``output_dir`` is then replaced as a whole (no stale
    models from earlier exports).

    Returns:
        Paths of the copied bundles

    Raises:
        FileNotFoundError: If a run lacks one of RUN_FILES
        ValueError: If two runs map to the same demo name
    """
    plan = [parse_run(spec) for spec in runs]
    names = [name for _, name in plan]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Duplicate demo names: {duplicates}")

    missing = [
        str(artifacts_dir / source / filename)
        for source, _ in plan for filename in RUN_FILES
        if not (artifacts_dir / source / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing files (train these runs first):\n  "
            + "\n  ".join(missing)
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    bundles = []
    for source, name in plan:
        target = output_dir / name
        target.mkdir(parents=True)
        for filename in RUN_FILES:
            shutil.copy2(artifacts_dir / source / filename, target / filename)
        bundles.append(target / RUN_FILES[0])
    return bundles


def main(args: argparse.Namespace) -> int:
    try:
        bundles = export_runs(
            args.runs, _resolve(args.artifacts_dir), _resolve(args.output_dir)
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    total_kb = sum(b.stat().st_size for b in bundles) / 1024
    for bundle in bundles:
        print(f"  {bundle.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Exported {len(bundles)} models ({total_kb:.0f} KB of bundles)")
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
