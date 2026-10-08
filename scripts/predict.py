#!/usr/bin/env python3
"""
Prediction script for Airline Sentiment Analysis.

Usage:
    # Single text prediction, JSON output
    python scripts/predict.py \
        --model artifacts/model_bundle.joblib \
        --text "Great flight!" \
        --output json

    # Batch prediction, CSV output with explanations
    python scripts/predict.py \
        --model artifacts/model_bundle.joblib \
        --input data/new_tweets.csv \
        --output csv \
        --explain

    # Quiet mode, auto-save to artifacts/predict/
    python scripts/predict.py \
        --model artifacts/model_bundle.joblib \
        --input data/batch.csv \
        --output json \
        --quiet

Output: Predictions auto-saved to
artifacts/predict/predict_YYYYMMDD_HHMMSS.{json|csv}
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd
import matplotlib

# Use non-interactive backend for saving plots without display
matplotlib.use("Agg")

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.interpretability import (  # noqa: E402
    explain_prediction,
    SHAP_AVAILABLE,
    supports_word_explanations,
)
from src.models import (  # noqa: E402
    load_model,
    predict_sentiment
)
from src.utils.logging_config import setup_logger  # noqa: E402

logger = setup_logger("predict", level="INFO")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for prediction script.

    Returns:
        argparse.Namespace with parsed CLI options

    Note:
        --text and --input are mutually exclusive (argparse handles this)
        --explain generates interpretable examples (slower, for analysis)
        --quiet suppresses stdout (useful for CI/automation)
    """
    parser = argparse.ArgumentParser(
        description="Predict sentiment for airline tweets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model bundle (.joblib file)",
    )

    # Mutually exclusive group:
    # user must provide either single text or batch file
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text to predict",
    )
    input_group.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to CSV file with 'text' column for batch prediction",
    )

    parser.add_argument(
        "--output",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Output format for predictions (default: json)",
    )

    # Explanation flags: generate interpretable examples (debug/analysis only)
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate explanations for predictions (top contributors)",
    )

    parser.add_argument(
        "--use-shap",
        action="store_true",
        help="Use SHAP for explanations (requires shap package)",
    )

    parser.add_argument(
        "--n-explain",
        type=int,
        default=5,
        help="Number of top contributing words to show in explanations",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress console output "
            "(results always saved to artifacts/predict/)"
        ),
    )

    return parser.parse_args()


def load_model_bundle(model_path: str | Path):
    """
    Load trained model, vectorizer, and mappings from joblib bundle.

    Args:
        model_path: Path to .joblib file containing model bundle

    Returns:
        Tuple of (model, vectorizer, target_mapping_inv)

    Raises:
        FileNotFoundError: If bundle doesn't exist at specified path
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")

    model, vectorizer, target_mapping, target_mapping_inv = load_model(
        model_path
    )
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Classes: {list(target_mapping_inv.values())}")
    return model, vectorizer, target_mapping_inv


def predict_single(
    model,
    vectorizer,
    text: str,
    class_names: list[str] | None = None,
    explain: bool = False,
    use_shap: bool = False,
    n_explain: int = 5,
) -> dict:
    """
    Predict sentiment for a single text with optional explanation.

    Args:
        model: Loaded LogisticRegression model
        vectorizer: Loaded vectorizer (TfidfVectorizer or CountVectorizer)
        text: Input text to classify
        class_names: Optional list of class names for readable output
        explain: Whether to generate word-level explanation
        use_shap: Whether to use SHAP instead of model weights
        n_explain: Number of top contributors to include

    Returns:
        Dict with prediction, probabilities, timestamp,
        and optional explanation
    """
    # Get prediction with probabilities (return_proba=True)
    pred_idx, proba = predict_sentiment(
        model, vectorizer, text, return_proba=True
    )
    pred_idx = int(pred_idx[0])
    proba = proba[0]

    # Decode integer prediction to human-readable label
    pred_label = class_names[pred_idx]

    # Build result dict with all required fields for API compatibility
    result = {
        "text": text,
        "predicted_class": pred_label,
        "predicted_class_idx": pred_idx,
        "probabilities": {
            (
                class_names[i]
                if class_names and i < len(class_names)
                else str(i)
            ): float(p)
            for i, p in enumerate(proba)
        },
        "confidence": float(proba.max()),
        "timestamp": datetime.now().isoformat(),
    }

    # Add explanation if requested (local interpretability);
    # not available for dense sentence embeddings
    if explain and supports_word_explanations(vectorizer):
        exp_result = explain_prediction(
            model,
            vectorizer,
            text,
            class_names=class_names,
            n_top_contributors=n_explain,
            use_shap=use_shap and SHAP_AVAILABLE,
        )
        result["explanation"] = {
            "method": exp_result["method"],
            "top_contributors": exp_result["top_contributors"],
        }

    return result


def predict_batch(
    model,
    vectorizer,
    texts: list[str],
    class_names: list[str] | None = None,
    explain: bool = False,
    use_shap: bool = False,
    n_explain: int = 5,
) -> list[dict]:
    """
    Predict sentiment for multiple texts sequentially.

    Args:
        model: Loaded LogisticRegression model
        vectorizer: Loaded vectorizer
        texts: List of input texts to classify
        class_names: Optional list of class names for readable output
        explain: Whether to generate explanations for all predictions
        use_shap: Whether to use SHAP for explanations
        n_explain: Number of top contributors per explanation

    Returns:
        List of prediction dicts (same length as valid input texts)

    Note:
        Invalid texts (empty/non-string) are skipped with warning
        Progress logged every 10 items for large batches
    """
    results = []
    for i, text in enumerate(texts):
        # Skip invalid inputs
        if not isinstance(text, str) or not text.strip():
            logger.warning(f"Skipping invalid text at index {i}")
            continue

        result = predict_single(
            model, vectorizer, text,
            class_names=class_names,
            explain=explain,
            use_shap=use_shap,
            n_explain=n_explain,
        )
        results.append(result)

        # Progress logging for large batches
        if len(texts) > 10 and (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(texts)} texts")

    return results


def save_results(results: list[dict], output_path: str | Path) -> None:
    """
    Save predictions to JSON or CSV based on file extension.

    Args:
        results: List of prediction dicts from predict_batch()
        output_path: Path to output file (.json or .csv)

    Raises:
        ValueError: If output format is not .json or .csv

    Note:
        CSV export flattens nested structures (probabilities, explanations)
        JSON export preserves full nested structure for API compatibility
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        # JSON: preserve full nested structure (API-compatible)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.debug(f"Predictions saved to {output_path}")

    elif output_path.suffix.lower() == ".csv":
        # CSV: flatten nested structures for spreadsheet compatibility
        flat_results = []
        for r in results:
            row = {
                "text": r["text"],
                "predicted_class": r["predicted_class"],
                "confidence": r["confidence"],
                "timestamp": r["timestamp"],
            }
            # Add probabilities as separate columns
            for label, prob in r["probabilities"].items():
                row[f"prob_{label}"] = prob
            # Add explanation if present
            if "explanation" in r:
                row["explanation_method"] = r["explanation"]["method"]
                top_contribs = r["explanation"]["top_contributors"][:3]
                row["top_contributors"] = "; ".join(
                    [f"{w}({wt:+.2f})" for w, wt in top_contribs]
                )
            flat_results.append(row)

        pd.DataFrame(flat_results).to_csv(
            output_path, index=False, encoding="utf-8"
        )
        logger.debug(f"Predictions saved to {output_path}")

    else:
        # Explicit error for unsupported formats
        raise ValueError(
            f"Unsupported output format: {output_path.suffix}. "
            "Use .json or .csv"
        )


def main(args: argparse.Namespace) -> int:
    """
    Main prediction pipeline.

    Args:
        args: Parsed CLI arguments from parse_args()

    Returns:
        Exit code (0=success, 1=error)

    Pipeline:
        1. Load model bundle
        2. Prepare input texts (single or batch)
        3. Run predictions with optional explanations
        4. Output results to console (if not --quiet)
        5. Save results to file (auto-generated path)
        6. Print summary
    """
    start_time = datetime.now()

    # Load model bundle with error handling
    try:
        model, vectorizer, target_mapping_inv = load_model_bundle(args.model)
        # Build ordered class names for consistent indexing
        class_names = [
            target_mapping_inv.get(idx, str(idx))
            for idx in sorted(target_mapping_inv.keys())
        ]
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Prepare input: either single text or batch from CSV
    if args.text:
        texts = [args.text]
        logger.info("Predicting for 1 text")
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1

        df = pd.read_csv(input_path)
        if "text" not in df.columns:
            logger.error(
                "Input CSV must contain 'text' column. "
                f"Found: {list(df.columns)}"
            )
            return 1

        # Drop NaN and ensure all texts are strings
        texts = df["text"].dropna().astype(str).tolist()
        logger.info(f"Loaded {len(texts)} texts from {input_path}")

    if args.explain and not supports_word_explanations(vectorizer):
        logger.warning(
            "--explain ignored: word-level explanations are not supported "
            f"for {type(vectorizer).__name__}"
        )

    # Run predictions (sequential, with optional explanations)
    logger.info("Running predictions...")
    results = predict_batch(
        model,
        vectorizer,
        texts,
        class_names=class_names,
        explain=args.explain,
        use_shap=args.use_shap,
        n_explain=args.n_explain,
    )

    # Print preview to console (if not --quiet) for immediate feedback
    if not args.quiet:
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        for r in results[:5]:  # Show first 5
            text_preview = (
                r['text'][:80] + ('...' if len(r['text']) > 80 else '')
            )
            print(f"\nText: {text_preview}")
            print(
                f"Predicted: {r['predicted_class']} "
                f"(confidence: {r['confidence']:.2f})"
            )
            if "explanation" in r:
                top_contribs = r["explanation"]["top_contributors"][:3]
                contributors = ", ".join(
                    [f"{w}({wt:+.2f})" for w, wt in top_contribs]
                )
                print(f"Top contributors: {contributors}")
        if len(results) > 5:
            print(f"\n... and {len(results) - 5} more predictions")
        print("=" * 60 + "\n")

    output_path = None

    # Auto-save results to artifacts/predict/ with timestamped filename
    if args.output:
        output_dir = Path(PROJECT_ROOT) / "artifacts" / "predict"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-generate filename: predict_YYYYMMDD_HHMMSS.{format}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predict_{timestamp}.{args.output}"
        output_path = output_dir / filename

        try:
            save_results(results, output_path)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return 1

    # Summary logging and optional stdout output
    elapsed = datetime.now() - start_time
    logger.info(f"Prediction completed in {elapsed}")
    logger.info(f"Processed {len(results)} texts")

    if not args.quiet:
        print(f"Processed {len(results)} texts in {elapsed}")
        if args.output and output_path is not None:
            try:
                relative_path = output_path.relative_to(PROJECT_ROOT)
                print(f"Results saved to: {relative_path}")
            except ValueError:
                print(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    args = parse_args()
    exit_code = main(args)
    sys.exit(exit_code)
