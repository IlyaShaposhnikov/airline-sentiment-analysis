#!/usr/bin/env python3
"""Integration test for the full ML pipeline."""
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_config
from src.preprocessing import create_vectorizer, preprocess_texts
from src.models import (
    train_model, evaluate_model, prepare_sample_weights,
    save_model, load_model, predict_sentiment, decode_predictions,
    save_evaluation_results
)
from src.utils.logging_config import setup_logger

logger = setup_logger("test_pipeline", level="INFO")


def generate_synthetic_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate simple synthetic tweet-like data for testing."""
    np.random.seed(42)
    
    templates = {
        0: ["terrible service", "worst flight ever", "never again", "so disappointed"],
        1: ["great flight", "loved it", "excellent service", "highly recommend"],
        2: ["okay experience", "nothing special", "average flight", "meh"]
    }
    
    texts = []
    labels = []
    confidences = []
    
    for _ in range(n_samples):
        label = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])  # Imbalanced
        text = np.random.choice(templates[label])
        # Add some noise
        if np.random.random() > 0.7:
            text += " " + np.random.choice(["!!!", "???", "ok", "yeah"])
        texts.append(text)
        labels.append(label)
        confidences.append(np.random.uniform(0.7, 1.0))
    
    return pd.DataFrame({
        "airline_sentiment": ["negative", "positive", "neutral"][np.array(labels)],
        "text": texts,
        "airline_sentiment_confidence": confidences,
        "negativereason_confidence": np.random.uniform(0.5, 1.0, n_samples),
    })


def main():
    logger.info("🚀 Starting integration test...")
    
    # 1. Load config
    cfg = load_config()
    logger.info("✅ Config loaded")
    
    # 2. Generate or load data
    try:
        from src.data_loader import load_and_prepare_data
        df = load_and_prepare_data()
        logger.info(f"✅ Loaded real data: {len(df)} rows")
    except FileNotFoundError:
        logger.warning("⚠️ Real data not found, using synthetic data")
        df = generate_synthetic_data()
    
    # 3. Prepare features and target
    texts = df["text"].tolist()
    y = df["target"].values
    confidences = df["sentiment_confidence"].values
    
    # 4. Preprocess and vectorize
    vectorizer = create_vectorizer(cfg)
    texts_processed = preprocess_texts(texts, cfg)
    X = vectorizer.fit_transform(texts_processed)
    logger.info(f"✅ Vectorized: {X.shape}")
    
    # 5. Train/test split
    X_train, X_test, y_train, y_test, weights_train, _ = train_test_split(
        X, y, confidences,
        test_size=cfg.get("test_size", 0.25),
        random_state=cfg.get("random_state", 42),
        stratify=y
    )
    
    # 6. Prepare sample weights
    sample_weights = prepare_sample_weights(
        pd.DataFrame({"conf": weights_train}), "conf", normalize=False
    )
    
    # 7. Train model
    model = train_model(X_train, y_train, cfg, sample_weights=sample_weights)
    logger.info("✅ Model trained")
    
    # 8. Evaluate
    metrics = evaluate_model(model, X_test, y_test, cfg, labels=[0, 1, 2])
    logger.info(f"✅ Evaluation complete: accuracy={metrics['accuracy']:.3f}")
    
    # 9. Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = save_model(model, vectorizer, tmpdir)
        logger.info(f"✅ Model saved to {model_path}")
        
        # 10. Load and predict
        loaded_model, loaded_vec, mapping, mapping_inv = load_model(model_path)
        
        test_texts = ["great flight!", "terrible service", "meh"]
        preds, probas = predict_sentiment(loaded_model, loaded_vec, test_texts, return_proba=True)
        labels = decode_predictions(preds, mapping_inv)
        
        for txt, label, prob in zip(test_texts, labels, probas):
            logger.info(f"  '{txt}' → {label} (confidence: {prob.max():.2f})")
        
        # 11. Save metrics
        metrics_path = save_evaluation_results(metrics, tmpdir)
        logger.info(f"✅ Metrics saved to {metrics_path}")
    
    logger.info("🎉 Integration test PASSED!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
