#!/usr/bin/env python3
"""
Compare feature extractors x classifiers on identical splits.

Grid (any subset via --configs):
    tfidf_lr        TF-IDF            + LogisticRegression
    tfidf_mlp       TF-IDF            + PyTorch MLP
    embedding_lr    Sentence embedding + LogisticRegression
    embedding_mlp   Sentence embedding + PyTorch MLP

For every seed, all configurations share the same stratified split and the
same confidence weights, so differences come from the representation and
the classifier only. Results are reported as mean ± std over seeds.

Sentence embeddings are computed ONCE for the whole dataset and cached on
disk (the encoder is pretrained and frozen, so encoding is independent of
the split; nothing is learned from test rows). Reruns reuse the cache.

Usage:
    python scripts/compare.py
    python scripts/compare.py --seeds 42 43 44 \
        --configs tfidf_lr embedding_lr
    python scripts/compare.py --binary-mode \
        --output-dir artifacts/compare_binary

Outputs (in --output-dir, default artifacts/compare):
    results.csv   one row per (configuration, seed)
    summary.csv   mean/std per configuration
    summary.md    Markdown table, ready for the README
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: import src (any submodule) before scikit-learn (see src/__init__.py)
from src.data_loader import (  # noqa: E402
    load_config,
    load_and_prepare_data,
    split_train_test_indices,
)
from src.models import (  # noqa: E402
    predict_sentiment,
    prepare_sample_weights,
    train_model,
)
from src.preprocessing import create_vectorizer  # noqa: E402
from src.utils.logging_config import setup_logger  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    roc_auc_score,
)

logger = setup_logger("compare", level="INFO")

CONFIGS: dict[str, tuple[str, str]] = {
    "tfidf_lr": ("tfidf", "logistic_regression"),
    "tfidf_mlp": ("tfidf", "mlp"),
    "embedding_lr": ("sentence_embedding", "logistic_regression"),
    "embedding_mlp": ("sentence_embedding", "mlp"),
}

LABELS = {
    "tfidf_lr": "TF-IDF + LogReg",
    "tfidf_mlp": "TF-IDF + MLP",
    "embedding_lr": "Embeddings + LogReg",
    "embedding_mlp": "Embeddings + MLP",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF/embeddings x LogReg/MLP on equal splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="artifacts/compare")
    parser.add_argument(
        "--cache-dir", default="artifacts/cache",
        help="Where sentence embeddings are cached",
    )
    parser.add_argument(
        "--configs", nargs="+", choices=list(CONFIGS),
        default=list(CONFIGS), help="Configurations to run",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46],
        help="One split + model seed per value",
    )
    parser.add_argument(
        "--binary-mode", action="store_true",
        help="Positive/negative only (exclude neutral)",
    )
    parser.add_argument(
        "--latency-samples", type=int, default=200,
        help="Test texts used to measure inference latency",
    )
    return parser.parse_args(argv)


# ============================================================================
# Embedding cache
# ============================================================================


def _embedding_cache_key(vectorizer, texts: list[str]) -> str:
    """Hash of encoder settings + light-cleaning settings + exact texts."""
    params = {
        k: v for k, v in vectorizer.get_params().items()
        if k != "preprocessor"
    }
    preprocessor = vectorizer.preprocessor
    cleaning = getattr(preprocessor, "keywords", None)
    h = hashlib.sha256()
    h.update(json.dumps(
        {"params": params, "cleaning": cleaning}, sort_keys=True, default=str
    ).encode("utf-8"))
    for text in texts:
        h.update(b"\x00")
        h.update(str(text).encode("utf-8"))
    return h.hexdigest()[:20]


def encode_with_cache(
    vectorizer, texts: list[str], cache_dir: Path
) -> tuple[np.ndarray, float | None]:
    """
    Encode ``texts`` once; reuse ``cache_dir/emb_<hash>.npy`` on reruns.

    Returns:
        (embeddings, encoding_seconds or None if loaded from cache)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"emb_{_embedding_cache_key(vectorizer, texts)}.npy"
    if path.exists():
        logger.info(f"Loaded cached embeddings: {path}")
        return np.load(path, allow_pickle=False), None

    logger.info(f"Encoding {len(texts)} texts (cached afterwards)...")
    start = time.perf_counter()
    embeddings = vectorizer.transform(texts)
    seconds = time.perf_counter() - start
    np.save(path, embeddings, allow_pickle=False)
    logger.info(f"Encoded in {seconds:.1f}s, saved to {path}")
    return embeddings, seconds


# ============================================================================
# Metrics and latency
# ============================================================================


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    if y_proba.shape[1] == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:  # same definition as src/metrics.py: one-vs-one, macro
        auc = roc_auc_score(y_true, y_proba, multi_class="ovo",
                            average="macro")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "roc_auc": auc,
    }


def measure_latency(model, vectorizer, texts: list[str]) -> dict[str, float]:
    """
    End-to-end inference cost from RAW text (cleaning + vectorization +
    model), as the API pays it.
    """
    predict_sentiment(model, vectorizer, texts[:2])  # warm-up

    start = time.perf_counter()
    predict_sentiment(model, vectorizer, texts)
    batch_ms = (time.perf_counter() - start) * 1000 / len(texts)

    single = []
    for text in texts[:30]:
        start = time.perf_counter()
        predict_sentiment(model, vectorizer, text)
        single.append((time.perf_counter() - start) * 1000)

    return {
        "latency_batch_ms": batch_ms,
        "latency_single_ms": float(np.median(single)),
    }


# ============================================================================
# Main
# ============================================================================


def run_config(
    name: str, base_cfg: dict, df: pd.DataFrame, seed: int,
    embeddings: np.ndarray | None, embedding_vectorizer,
    latency_samples: int, measure: bool,
) -> dict:
    vec_type, model_type = CONFIGS[name]
    cfg = copy.deepcopy(base_cfg)
    cfg["preprocessing"].setdefault("vectorizer", {})["type"] = vec_type
    cfg["model"]["type"] = model_type
    cfg["model"].setdefault("training", {})["random_state"] = seed

    y = df["target"].to_numpy()
    train_idx, test_idx = split_train_test_indices(y, cfg)
    texts = df["text"].tolist()
    y_train, y_test = y[train_idx], y[test_idx]

    if vec_type == "sentence_embedding":
        vectorizer = embedding_vectorizer
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    else:  # fitted on the training rows of THIS seed only
        vectorizer = create_vectorizer(cfg)
        X_train = vectorizer.fit_transform([texts[i] for i in train_idx])
        X_test = vectorizer.transform([texts[i] for i in test_idx])

    sample_weights = None
    if cfg["model"]["training"].get("use_confidence_weights", True):
        w = df["sentiment_confidence"].to_numpy()[train_idx]
        sample_weights = prepare_sample_weights(
            pd.DataFrame({"conf": w}), "conf", normalize=False
        )

    start = time.perf_counter()
    model = train_model(X_train, y_train, cfg, sample_weights=sample_weights)
    fit_seconds = time.perf_counter() - start

    y_proba = model.predict_proba(X_test)
    y_pred = model.classes_[np.argmax(y_proba, axis=1)]

    row = {
        "config": name, "seed": seed, "n_features": X_train.shape[1],
        **compute_metrics(y_test, y_pred, y_proba),
        "fit_seconds": fit_seconds,
    }
    if measure:
        sample = [texts[i] for i in test_idx[:latency_samples]]
        row.update(measure_latency(model, vectorizer, sample))
    return row


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "f1_macro", "roc_auc", "fit_seconds"]
    summary = results.groupby("config", sort=False)[metrics].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    for col in ["latency_batch_ms", "latency_single_ms", "n_features"]:
        if col in results:
            summary[col] = results.groupby("config", sort=False)[col].first()
    summary["n_seeds"] = results.groupby("config", sort=False).size()
    return summary.reset_index()


def to_markdown(summary: pd.DataFrame, n_seeds: int, mode: str) -> str:
    def pm(row, metric):
        std = row[f"{metric}_std"]
        std = 0.0 if pd.isna(std) else std
        return f"{row[f'{metric}_mean']:.3f} ± {std:.3f}"

    best = {m: summary[f"{m}_mean"].max()
            for m in ["accuracy", "f1_macro", "roc_auc"]}
    lines = [
        f"| Configuration ({mode}) | Accuracy | F1 (macro) | ROC-AUC "
        "| Fit, s | Latency, ms/text (batch / single) |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in summary.iterrows():
        cells = []
        for m in ["accuracy", "f1_macro", "roc_auc"]:
            cell = pm(row, m)
            cells.append(
                f"**{cell}**" if row[f"{m}_mean"] == best[m] else cell
            )
        latency = (
            f"{row['latency_batch_ms']:.2f} / "
            f"{row['latency_single_ms']:.1f}"
            if "latency_batch_ms" in row and not pd.isna(
                row["latency_batch_ms"]
            ) else "—"
        )
        lines.append(
            f"| {LABELS[row['config']]} | " + " | ".join(cells)
            + f" | {row['fit_seconds_mean']:.1f} | {latency} |"
        )
    lines.append("")
    lines.append(
        f"Mean ± std over {n_seeds} stratified splits (seeds); best mean in "
        "bold. Fit time excludes one-off embedding encoding (cached). "
        "Latency is end-to-end from raw text on CPU."
    )
    return "\n".join(lines) + "\n"


def main(args: argparse.Namespace) -> int:
    started = datetime.now()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    base_cfg = load_config(config_path)
    base_cfg.setdefault("preprocessing", {})
    base_cfg.setdefault("model", {})

    df = load_and_prepare_data(config_path=config_path, base_dir=PROJECT_ROOT)
    df = df.reset_index(drop=True)

    embeddings = embedding_vectorizer = encoding_seconds = None
    n_encoded = len(df)
    if any(CONFIGS[c][0] == "sentence_embedding" for c in args.configs):
        emb_cfg = copy.deepcopy(base_cfg)
        emb_cfg["preprocessing"].setdefault("vectorizer", {})["type"] = (
            "sentence_embedding"
        )
        embedding_vectorizer = create_vectorizer(emb_cfg)
        try:
            # Encode the full (pre-binary-filter) dataset: one cache entry
            # serves both multiclass and binary comparisons
            embeddings, encoding_seconds = encode_with_cache(
                embedding_vectorizer, df["text"].tolist(),
                Path(args.cache_dir) if Path(args.cache_dir).is_absolute()
                else PROJECT_ROOT / args.cache_dir,
            )
            embedding_vectorizer.fit(["dimension probe"])
        except ImportError as e:
            logger.warning(f"Skipping embedding configurations: {e}")
            args.configs = [
                c for c in args.configs
                if CONFIGS[c][0] != "sentence_embedding"
            ]

    if args.binary_mode:
        keep = df["target"].isin([0, 1]).to_numpy()
        df = df[keep].reset_index(drop=True)
        if embeddings is not None:
            embeddings = embeddings[keep]

    rows = []
    for name in args.configs:
        for i, seed in enumerate(args.seeds):
            logger.info(f"=== {name}, seed {seed} ===")
            try:
                rows.append(run_config(
                    name, base_cfg, df, seed, embeddings,
                    embedding_vectorizer, args.latency_samples,
                    measure=(i == 0),
                ))
            except ImportError as e:
                logger.warning(f"Skipping {name}: {e}")
                break

    if not rows:
        logger.error("No configuration could be run")
        return 1

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results.csv", index=False)
    summary = summarize(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    mode = "binary" if args.binary_mode else "3 classes"
    markdown = to_markdown(summary, len(args.seeds), mode)
    if encoding_seconds is not None:
        markdown += (
            f"\nOne-off embedding encoding of {n_encoded} texts: "
            f"{encoding_seconds:.0f}s.\n"
        )
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")

    print("\n" + markdown)
    print(f"Saved to {output_dir} in {datetime.now() - started}")
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
