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
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

# Use non-interactive backend for saving plots without display
matplotlib.use("Agg")

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config, load_and_prepare_data  # noqa: E402
from src.interpretability import (  # noqa: E402
    explain_prediction,
    get_top_features_by_weight,
    plot_feature_importance,
    SHAP_AVAILABLE,
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
from src.preprocessing import create_vectorizer, preprocess_texts  # noqa: E402
from src.utils.logging_config import setup_logger  # noqa: E402

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
        3. Preprocess texts and vectorize
        4. Train/test split (stratified)
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
    # 3. Preprocess texts and vectorize
    # =========================================================================
    logger.info("Preprocessing texts...")
    vectorizer = create_vectorizer(cfg)
    texts_processed = preprocess_texts(df["text"].tolist(), cfg)

    # fit_transform learns vocabulary from training data only
    # (prevents data leakage)
    X = vectorizer.fit_transform(texts_processed)
    y = df["target"].values
    logger.info(f"Vectorized: {X.shape[0]} samples × {X.shape[1]} features")

    # =========================================================================
    # 4. Train/test split (stratified)
    # =========================================================================
    logger.info("Splitting data...")

    eval_cfg = cfg.get("evaluation", {})
    split_cfg = eval_cfg.get("split", {})
    model_cfg = cfg.get("model", {})
    training_cfg = model_cfg.get("training", {})
    stratify_enabled = split_cfg.get("stratify", True)
    # Stratify preserves class distribution in train/test splits
    # (critical for imbalanced data)
    stratify_value = y if stratify_enabled else None

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        df["sentiment_confidence"].values,
        test_size=split_cfg.get("test_size", 0.25),
        random_state=training_cfg.get("random_state", 42),
        stratify=stratify_value,
    )
    logger.info(f"Split: train={X_train.shape[0]}, test={X_test.shape[0]}")

    if X_test.shape[0] == 0:
        logger.error(
            "Test set is empty after split. "
            f"Check test_size={split_cfg.get('test_size', 0.25)} "
            f"and data size={len(X)}."
        )
        return 1

    # Extract original texts for test set to enable explanations
    # and misclassified analysis
    test_texts = df["text"].iloc[-X_test.shape[0]:].tolist()

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
        plot_settings = interp_cfg.get("plot_settings", {})

        for class_idx, class_name in enumerate(class_names):
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

        logger.info(f"Saved {len(class_names) + 1} plots to {output_dir}")

    # =========================================================================
    # 8. Interpretability: top features and prediction explanations
    # =========================================================================
    if args.explain:
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
    print(f"Features: {X.shape[1]}")
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
