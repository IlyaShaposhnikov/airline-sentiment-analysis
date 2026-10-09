#!/usr/bin/env python3
"""
Main training script for Airline Sentiment Analysis.

Usage:
    python scripts/train.py                          # Run with default config
    python scripts/train.py --config configs/v2.yaml # Custom config
    python scripts/train.py --binary-mode            # Binary classification
    python scripts/train.py --no-plots               # Skip visualization
    python scripts/train.py --explain --n-explain 5  # Explain N predictions

Output artifacts are saved to artifacts/ directory.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import warnings

import matplotlib
import pandas as pd
from scipy.sparse import issparse

# Use non-interactive backend for saving plots without display
matplotlib.use("Agg")

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: import src (any submodule) before scikit-learn: on Windows the
# package preloads torch to avoid an OpenMP runtime conflict
# (see src/__init__.py)
from src.data_loader import (  # noqa: E402
    load_config,
    load_and_prepare_data,
    split_train_test_indices,
)
from src.interpretability import (  # noqa: E402
    explain_prediction,
    get_top_features_by_weight,
    plot_feature_importance,
    SHAP_AVAILABLE,
    supports_word_explanations,
)
from src.metrics import (  # noqa: E402
    compute_comprehensive_metrics,
    export_metrics,
    get_top_misclassified,
    plot_confusion_matrix,
)
from src.models import (  # noqa: E402
    prepare_sample_weights,
    save_model,
    train_model,
)
from src.preprocessing import create_vectorizer  # noqa: E402
from src.utils.logging_config import setup_logger  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

# Configure root logger
logger = setup_logger("train", level="INFO", log_file="artifacts/training.log")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training script.

    Returns:
        argparse.Namespace with parsed CLI options

    Note:
        --binary-mode excludes neutral class (target=2)
        --explain generates sample explanations (slow, for debugging)
        --use-shap requires shap package installed
    """
    parser = argparse.ArgumentParser(
        description="Train airline sentiment classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to save model and reports",
    )

    # Binary mode: filter out neutral class for simpler classification task
    parser.add_argument(
        "--binary-mode",
        action="store_true",
        help=(
            "Train binary classifier (positive/negative only, exclude neutral)"
        ),
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (faster execution)",
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate explanations for sample predictions",
    )

    # Explanation flags: generate interpretable examples (debug/analysis only)
    parser.add_argument(
        "--n-explain",
        type=int,
        default=5,
        help="Number of predictions to explain (when --explain is used)",
    )

    parser.add_argument(
        "--use-shap",
        action="store_true",
        help="Use SHAP for explanations (requires shap package)",
    )

    # Vectorizer override: switch feature extraction without editing config
    parser.add_argument(
        "--vectorizer",
        type=str,
        choices=["tfidf", "count", "sentence_embedding"],
        default=None,
        help=(
            "Override preprocessing.vectorizer.type from config "
            "(sentence_embedding requires requirements-dl.txt)"
        ),
    )

    # Seed override: useful for reproducibility experiments
    # without editing config
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random_state from config (for reproducibility)",
    )

    return parser.parse_args()


def filter_binary_data(
    df: pd.DataFrame, target_col: str = "target"
) -> pd.DataFrame:
    """
    Filter DataFrame to keep only positive (1) and negative (0) samples.

    Args:
        df: Input DataFrame with encoded target column
        target_col: Name of target column (default: "target")

    Returns:
        Filtered DataFrame with only binary classes

    Note:
        Used when --binary-mode flag is passed to training script
    """
    df_binary = df[df[target_col].isin([0, 1])].copy()
    logger.info(
        f"Binary mode: filtered from {len(df)} to {len(df_binary)} samples "
        f"(removed neutral class)"
    )
    return df_binary


def main(args: argparse.Namespace) -> int:
    """
    Main training pipeline.

    Args:
        args: Parsed CLI arguments from parse_args()

    Returns:
        Exit code (0=success, 1=error)

    Pipeline stages:
        1. Load configuration
        2. Load and prepare data
        3. Train/test split (stratified, by row position)
        4. Fit vectorizer on train texts only, transform test
        5. Train model
        6. Evaluate model
        7. Generate visualizations and reports
        8. Interpretability: top features and explanations
        9. Export metrics and save model
        10. Summary and exit
    """
    start_time = datetime.now()
    logger.info(
        f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # =========================================================================
    # 1. Load configuration
    # =========================================================================
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # CLI seed override takes precedence over config
    # for reproducibility experiments
    if args.seed is not None:
        if "model" not in cfg:
            cfg["model"] = {}
        if "training" not in cfg["model"]:
            cfg["model"]["training"] = {}
        cfg["model"]["training"]["random_state"] = args.seed
        logger.info(f"Overridden random_state to {args.seed}")

    # CLI vectorizer override (getattr: main() is also called
    # programmatically with Namespaces that predate this flag)
    vectorizer_override = getattr(args, "vectorizer", None)
    if vectorizer_override:
        cfg.setdefault("preprocessing", {}).setdefault("vectorizer", {})
        cfg["preprocessing"]["vectorizer"]["type"] = vectorizer_override
        logger.info(f"Overridden vectorizer type to {vectorizer_override}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # =========================================================================
    # 2. Load and prepare data
    # =========================================================================
    logger.info("Loading data...")
    try:
        df = load_and_prepare_data(
            config_path=config_path, base_dir=PROJECT_ROOT
        )
    except FileNotFoundError as e:
        logger.error(f"Data loading failed: {e}")
        logger.error("Please download Tweets.csv to data/ directory")
        return 1

    logger.info(
        f"Data loaded: {len(df)} samples, "
        f"target distribution: {df['target'].value_counts().to_dict()}"
    )

    # Optional binary mode: exclude neutral class for simpler task
    if args.binary_mode:
        df = filter_binary_data(df)
        class_names = ["negative", "positive"]
        logger.info("Binary classification mode enabled")
        logger.info(
            "Binary mode distribution: "
            f"{df['target'].value_counts().to_dict()}"
        )
    else:
        class_names = ["negative", "positive", "neutral"]

    logger.info(
        f"Data prepared: {len(df)} samples, {len(df.columns)} columns"
    )

    # =========================================================================
    # 3. Train/test split (stratified)
    # =========================================================================
    # Split row positions first: texts, labels and confidence weights are then
    # selected with the same indices, so they stay aligned by construction.
    logger.info("Splitting data...")

    eval_cfg = cfg.get("evaluation", {})
    model_cfg = cfg.get("model", {})
    training_cfg = model_cfg.get("training", {})

    y = df["target"].to_numpy()
    try:
        train_idx, test_idx = split_train_test_indices(y, cfg)
    except ValueError as e:
        logger.error(f"{e}. Check evaluation.split.test_size in config.")
        return 1

    texts = df["text"].tolist()
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    w_train = df["sentiment_confidence"].to_numpy()[train_idx]
    logger.info(f"Split: train={len(train_idx)}, test={len(test_idx)}")

    # =========================================================================
    # 4. Vectorize (fit on train only — no test-set leakage into vocab/IDF)
    # =========================================================================
    # Text cleaning is embedded in the vectorizer (preprocessor=), so raw
    # texts go in here exactly as they will at inference time.
    logger.info("Vectorizing texts...")
    vectorizer = create_vectorizer(cfg)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    word_explanations = supports_word_explanations(vectorizer)
    logger.info(
        f"Vectorized: train {X_train.shape}, test {X_test.shape}, "
        f"vocabulary fitted on train only"
    )

    # Rows without any known feature usually mean over-aggressive cleaning
    # (only meaningful for sparse bag-of-words matrices)
    empty_train = (
        int((X_train.getnnz(axis=1) == 0).sum()) if issparse(X_train) else 0
    )
    if empty_train:
        logger.warning(
            f"{empty_train}/{X_train.shape[0]} training texts have no "
            "features after preprocessing/vectorization. "
            "Consider adjusting cleaning parameters."
        )

    # Confidence-based sample weights:
    # high-confidence annotations influence training more
    sample_weights = None
    if training_cfg.get("use_confidence_weights", True):
        sample_weights = prepare_sample_weights(
            pd.DataFrame({"conf": w_train}), "conf", normalize=False
        )
        logger.info("Using confidence-based sample weights")

    # =========================================================================
    # 5. Train model
    # =========================================================================
    logger.info("Training model...")
    model = train_model(X_train, y_train, cfg, sample_weights=sample_weights)
    logger.info("Model trained")

    # =========================================================================
    # 6. Evaluate model
    # =========================================================================
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # compute_comprehensive_metrics auto-detects binary/multiclass
    # and computes appropriate metrics
    reporting_cfg = eval_cfg.get("reporting", {})
    metrics = compute_comprehensive_metrics(
        y_test,
        y_pred,
        y_proba,
        config=reporting_cfg,
        class_names=class_names,
        auto_export=False,  # Export handled separately for more control
        output_dir=output_dir,
        export_filename="metrics",
    )

    # Log key metrics
    logger.info(
        f"Results: acc={metrics['accuracy']:.3f}, "
        f"f1={metrics['f1_macro']:.3f}, "
        f"auc={metrics.get('roc_auc', 'N/A')}"
    )

    # =========================================================================
    # 7. Generate visualizations and reports
    # =========================================================================
    interp_cfg = cfg.get("interpretability", {})
    weight_cfg = interp_cfg.get("weight_based", {})

    if not args.no_plots:
        logger.info("Generating visualizations...")

        # Confusion matrix
        cm_settings = reporting_cfg.get("plot_settings", {})
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            y_test,
            y_pred,
            tick_labels=class_names,
            normalize=cm_settings.get("normalize", True),
            cmap=cm_settings.get("cmap", "Blues"),
            figsize=tuple(cm_settings.get("figsize", [8, 6])),
            save_path=cm_path,
            title="Confusion Matrix (Test Set)",
        )

        # Feature importance per class:
        # shows which words drive predictions for each sentiment
        # (not applicable to dense embeddings: dimensions are not words)
        plot_settings = interp_cfg.get("plot_settings", {})
        word_plots = class_names if word_explanations else []
        if not word_explanations:
            logger.info(
                "Skipping feature-importance plots: not supported for "
                f"{type(vectorizer).__name__}"
            )

        for class_idx, class_name in enumerate(word_plots):
            feat_path = output_dir / f"feature_importance_{class_name}.png"
            plot_feature_importance(
                model,
                vectorizer,
                class_idx=class_idx,
                n_top=weight_cfg.get("n_top_features", 20),
                horizontal=plot_settings.get("horizontal", True),
                figsize=tuple(plot_settings.get("figsize", [10, 8])),
                save_path=feat_path,
                title=f"Top Features: {class_name}",
                class_names=class_names,
            )

        logger.info(f"Saved {len(word_plots) + 1} plots to {output_dir}")

    # =========================================================================
    # 8. Interpretability: top features and prediction explanations
    # =========================================================================
    if args.explain and not word_explanations:
        logger.warning(
            "--explain ignored: word-level explanations are not supported "
            f"for {type(vectorizer).__name__}"
        )
    elif args.explain:
        logger.info("Generating explanations...")

        # Top features by weight:
        # global model interpretability (not per-prediction)
        top_features = get_top_features_by_weight(
            model,
            vectorizer,
            n_top=weight_cfg.get("n_top_features", 20),
            threshold=weight_cfg.get("top_words_threshold", 2.0),
            class_names=class_names,
        )

        # Save top features to JSON
        features_path = output_dir / "top_features.json"
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(top_features, f, indent=2, ensure_ascii=False)
        logger.info(f"Top features saved to {features_path}")

        # Explain sample predictions:
        # local interpretability for debugging/analysis
        n_explain = min(args.n_explain, X_test.shape[0])
        explanations = []

        for i in range(n_explain):
            txt = test_texts[i]
            result = explain_prediction(
                model,
                vectorizer,
                txt,
                class_names=class_names,
                n_top_contributors=10,
                use_shap=args.use_shap and SHAP_AVAILABLE,
            )
            explanations.append(result)

        # Save explanations
        explanations_path = output_dir / "explanations.json"
        with open(explanations_path, "w", encoding="utf-8") as f:
            json.dump(explanations, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Saved {len(explanations)} explanations to {explanations_path}"
        )

        # Log sample explanations to console (quick feedback during training)
        logger.info("\nSample predictions:")
        for exp in explanations[:3]:  # Show first 3
            contributors = ", ".join(
                [f"{w}({wt:+.2f})" for w, wt in exp["top_contributors"][:3]]
            )
            logger.info(
                f"  '{exp['text'][:50]}...' → {exp['predicted_class']} | "
                f"{contributors}"
            )

    # =========================================================================
    # 9. Export metrics and save model
    # =========================================================================
    logger.info("Saving artifacts...")

    # Export metrics in configured formats (JSON/CSV)
    export_formats = reporting_cfg.get("export_formats", ["json", "csv"])
    export_metrics(
        metrics,
        output_dir,
        filename="metrics",
        formats=export_formats,
    )

    # Save top misclassified examples for manual review and model debugging
    misclassified_cfg = reporting_cfg.get("misclassified_examples", {})
    if misclassified_cfg.get("include_text", True):
        df_misclassified = get_top_misclassified(
            y_test,
            y_pred,
            y_proba,
            texts=test_texts,
            n_top=misclassified_cfg.get("n_top", 10),
            class_names=class_names,
        )
        if not df_misclassified.empty:
            misclassified_path = output_dir / "misclassified_examples.csv"
            df_misclassified.to_csv(misclassified_path, index=False)
            logger.info(
                f"Saved misclassified examples to {misclassified_path}"
            )

    # Save model bundle:
    # includes model, vectorizer, and label mappings for inference
    model_path = output_dir / "model_bundle.joblib"
    save_model(model, vectorizer, output_dir, filename="model_bundle.joblib")
    logger.info(f"Model saved to {model_path}")

    # Save config copy:
    # ensures reproducibility by capturing exact training settings
    config_backup = output_dir / "config_used.yaml"
    with open(config_backup, "w", encoding="utf-8") as f:
        import yaml

        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Config backup saved to {config_backup}")

    # =========================================================================
    # 10. Summary and exit
    # =========================================================================
    elapsed = datetime.now() - start_time
    logger.info(f"Training completed in {elapsed}")
    logger.info(f"Artifacts saved to: {output_dir}")

    artifact_files = list(Path(output_dir).glob("*"))
    logger.info(f"Total artifacts saved: {len(artifact_files)} files")
    for f in sorted(artifact_files):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            logger.debug(f"  - {f.name} ({size_kb:.1f} KB)")

    # Print quick summary to stdout for immediate feedback
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Mode: {'Binary' if args.binary_mode else 'Multiclass'}")
    print(
        f"Samples: {len(df)} "
        f"(train: {X_train.shape[0]}, test: {X_test.shape[0]})"
    )
    print(f"Features: {X_train.shape[1]}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 (macro): {metrics['f1_macro']:.3f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Artifacts: {output_dir}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    # Suppress sklearn convergence warnings for cleaner output
    # (not errors, just info)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    args = parse_args()
    exit_code = main(args)
    sys.exit(exit_code)
