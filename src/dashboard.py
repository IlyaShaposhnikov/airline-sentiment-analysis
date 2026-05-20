#!/usr/bin/env python3
"""
Streamlit dashboard for interactive sentiment analysis.

Provides a user-friendly web interface for:
- Single text prediction with confidence visualization
- Optional explanation toggle (top contributing words)
- Batch prediction via CSV upload/download
- Model status indicator

Usage:
    streamlit run src/dashboard.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.config import (  # noqa: E402
    MODEL_PATH, API_TITLE, API_VERSION, MAX_BATCH_SIZE
)
from src.api.services import model_service  # noqa: E402
from src.interpretability import SHAP_AVAILABLE  # noqa: E402
from src.utils.logging_config import setup_logger  # noqa: E402

# Configure logging (Streamlit captures stdout/stderr)
logger = setup_logger("dashboard", level="INFO")

# ============================================================================
# Initialize session state for explanation settings (persist across reloads)
# ============================================================================

if "single_explain" not in st.session_state:
    st.session_state.single_explain = False

if "single_shap" not in st.session_state:
    st.session_state.single_shap = False

if "single_n_explain" not in st.session_state:
    st.session_state.single_n_explain = 5

if "batch_explain" not in st.session_state:
    st.session_state.batch_explain = False

if "batch_shap" not in st.session_state:
    st.session_state.batch_shap = False

if "batch_n_explain" not in st.session_state:
    st.session_state.batch_n_explain = 5

# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title=API_TITLE,
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Header & status
# ============================================================================

st.title(f"✈️ {API_TITLE}")
st.caption(
    f"Version {API_VERSION} — Interactive sentiment analysis "
    "for airline tweets"
)


# Model status indicator
@st.cache_resource
def check_model_status() -> dict:
    """
    Check if model is available and loaded (cached to avoid repeated checks).
    """
    status = {
        "model_path": MODEL_PATH,
        "exists": Path(MODEL_PATH).exists(),
        "loaded": False,
        "shap_available": SHAP_AVAILABLE,
        "error": None,
    }
    try:
        model_service.load()
        status["loaded"] = True
    except Exception as e:
        status["error"] = str(e)
    return status


model_status = check_model_status()

# Status banner
if model_status["loaded"]:
    st.success("✅ Model loaded and ready")
elif model_status["exists"]:
    st.warning("⚠️ Model file found — loading on first prediction")
else:
    st.error(f"❌ Model not found at: {model_status['model_path']}")
    st.info(
        "💡 Run `python scripts/train.py` first to train and save the model"
    )

# Sidebar: info & settings
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Model Info")
    st.code(f"Path: {model_status['model_path']}", language="text")
    st.metric("Loaded", "Yes" if model_status["loaded"] else "No")
    shap_status = "Installed" if SHAP_AVAILABLE else "Not installed"
    st.metric("SHAP Library", shap_status)
    if SHAP_AVAILABLE:
        st.caption(
            "ℹ️ SHAP explanations may use zero baseline without training data"
        )
    st.metric("Model Version", API_VERSION)

    st.divider()

    st.subheader("Quick Tips")
    st.markdown("""
    - ✍️ Enter a tweet to see sentiment prediction
    - 🔍 Toggle "Explain" to see top contributing words
    - 📁 Upload CSV for batch prediction
    - 📊 Probabilities shown as horizontal bar chart
    """)

    st.divider()

    if st.button("🔄 Reload Model"):
        # Force re-check by clearing cache
        check_model_status.clear()
        st.rerun()

# ============================================================================
# Tabbed interface: Single / Batch / Info
# ============================================================================

tab_single, tab_batch, tab_info = st.tabs(
    ["🔍 Single Prediction", "📦 Batch Prediction", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# Tab 1: Single text prediction
# ---------------------------------------------------------------------------

with tab_single:
    st.header("Predict sentiment for a single tweet")

    # Explanation settings
    col1, col2 = st.columns([1, 2])
    with col1:
        explain_toggle = st.checkbox(
            "🔍 Show explanation",
            key="single_explain"
        )
    with col2:
        use_shap_toggle = st.checkbox(
            "Use SHAP (if available)",
            disabled=not SHAP_AVAILABLE,
            key="single_shap"
        )
        if not SHAP_AVAILABLE and use_shap_toggle:
            st.caption(
                "ℹ️ SHAP not installed — will fall back to model weights"
            )

    n_explain = st.slider(
        "Number of top words to show",
        min_value=1,
        max_value=20,
        disabled=not explain_toggle,
        key="single_n_explain"
    )

    # Text input + submit button
    with st.form("single_prediction_form"):
        text_input = st.text_area(
            "Enter tweet text",
            height=100,
            placeholder="@VirginAmerica Great flight, excellent service!",
            max_chars=1000,
            key="single_text_input"
        )

        submitted = st.form_submit_button(
            "🚀 Predict", type="primary", disabled=not model_status["exists"]
        )

    # Process prediction
    if submitted and text_input.strip():
        with st.spinner("Analyzing sentiment..."):
            try:
                # Ensure model is loaded
                model_service.load()

                # Run prediction via service layer
                response = model_service.predict_single_sync(
                    text=text_input.strip(),
                    explain=st.session_state.single_explain,
                    use_shap=st.session_state.single_shap and SHAP_AVAILABLE,
                    n_explain=st.session_state.single_n_explain,
                )

                # Display result
                st.subheader("📊 Result")

                # Prediction header
                col_prob, col_class = st.columns([2, 1])
                with col_class:
                    # Large sentiment badge
                    sentiment_color = {
                        "positive": "🟢",
                        "negative": "🔴",
                        "neutral": "🟡",
                    }.get(response.predicted_class, "⚪")
                    st.metric(
                        "Predicted Sentiment",
                        f"{sentiment_color} {response.predicted_class}",
                        f"Confidence: {response.confidence:.1%}",
                    )

                with col_prob:
                    # Probabilities as horizontal bar chart
                    prob_df = pd.DataFrame(
                        [
                            {"class": k, "probability": v}
                            for k, v in response.probabilities.items()
                        ]
                    ).sort_values("probability", ascending=False)

                    st.bar_chart(
                        prob_df.set_index("class"),
                        horizontal=True,
                        width='stretch',
                    )

                # Explanation section
                if explain_toggle and response.explanation:
                    st.subheader("🔍 Explanation")
                    st.caption(f"Method: `{response.explanation.method}`")

                    # Top contributors as table
                    contrib_df = pd.DataFrame(
                        response.explanation.top_contributors,
                        columns=["Word", "Contribution"]
                    )
                    contrib_df["Contribution"] = contrib_df[
                        "Contribution"
                    ].apply(lambda x: f"{x:+.3f}")

                    # Color-code contributions
                    def color_contrib(val):
                        if isinstance(val, str) and val.startswith("+"):
                            return "color: green"
                        elif isinstance(val, str) and val.startswith("-"):
                            return "color: red"
                        return ""

                    st.dataframe(
                        contrib_df.style.map(
                            color_contrib, subset=["Contribution"]
                        ),
                        width='stretch',
                        hide_index=True,
                    )

                # Raw JSON toggle (for developers)
                with st.expander("🔧 View raw response (JSON)"):
                    st.json(response.model_dump(mode='json'), expanded=False)

                logger.info(
                    f"Dashboard prediction: '{text_input[:50]}...' "
                    f"→ {response.predicted_class}"
                )

            except Exception as e:
                st.error(f"❌ Prediction failed: {type(e).__name__}: {e}")
                logger.error(f"Dashboard prediction error: {e}", exc_info=True)

    elif submitted and not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze")

# ---------------------------------------------------------------------------
# Tab 2: Batch prediction via CSV
# ---------------------------------------------------------------------------

with tab_batch:
    st.header("Predict sentiment for multiple tweets")

    st.markdown("""
    **Upload a CSV file** with a `text` column containing tweets.

    Example format:
    ```csv
    text
    "Great flight, thanks!"
    "Terrible delay, never again"
    "Okay experience, nothing special"
    ```
    """)

    uploaded_file = st.file_uploader(
        "📁 Choose CSV file", type=["csv"], disabled=not model_status["exists"]
    )

    if uploaded_file:
        # Read and preview
        try:
            df = pd.read_csv(uploaded_file)

            if len(df) > MAX_BATCH_SIZE:
                st.error(
                    f"❌ CSV too large: {len(df)} rows. "
                    f"Maximum allowed: {MAX_BATCH_SIZE}"
                )
                st.stop()

            if "text" not in df.columns:
                st.error(
                    "❌ CSV must contain a 'text' column. "
                    f"Found: {list(df.columns)}"
                )
            else:
                st.success(f"✅ Loaded {len(df)} rows")

                # Preview
                with st.expander("👀 Preview uploaded data"):
                    st.dataframe(df.head(), width='stretch')

                # Explanation settings
                col1, col2 = st.columns(2)
                with col1:
                    batch_explain = st.checkbox(
                        "🔍 Include explanations",
                        key="batch_explain"
                    )
                with col2:
                    batch_use_shap = st.checkbox(
                        "Use SHAP (if available)",
                        disabled=not SHAP_AVAILABLE,
                        key="batch_shap"
                    )

                batch_n_explain = st.slider(
                    "Top words per explanation",
                    min_value=1,
                    max_value=20,
                    disabled=not batch_explain,
                    key="batch_n_explain"
                )

                # Main batch form (only submit button)
                with st.form("batch_prediction_form"):
                    batch_submitted = st.form_submit_button(
                        "🚀 Run Batch Prediction", type="primary"
                    )

                if batch_submitted:
                    with st.spinner(f"Processing {len(df)} texts..."):
                        try:
                            model_service.load()

                            # Run predictions
                            texts = df["text"].dropna().astype(str).tolist()
                            results = []

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for i, text in enumerate(texts):
                                try:
                                    response = model_service.predict_single_sync(  # noqa: E501
                                        text=text,
                                        explain=st.session_state.batch_explain,
                                        use_shap=(
                                            st.session_state.batch_shap
                                            and SHAP_AVAILABLE
                                        ),
                                        n_explain=(
                                            st.session_state.batch_n_explain
                                        ),
                                    )
                                    results.append(response)
                                except Exception as e:
                                    logger.warning(
                                        f"Batch item {i} failed: {e}"
                                    )
                                    results.append(None)

                                # Update progress
                                progress_bar.progress((i + 1) / len(texts))
                                status_text.text(
                                    f"Processed {i + 1}/{len(texts)} texts"
                                )

                            status_text.text("✅ Done!")
                            progress_bar.empty()

                            failed_count = sum(1 for r in results if r is None)
                            if failed_count > 0:
                                st.warning(
                                    f"⚠️ {failed_count}/{len(texts)} "
                                    "predictions failed — "
                                    "check logs for details"
                                )

                            # Prepare output DataFrame
                            output_rows = []
                            for i, (text, resp) in enumerate(
                                zip(texts, results)
                            ):
                                if resp is None:
                                    output_rows.append({
                                        "index": i,
                                        "text": text,
                                        "error": "Prediction failed",
                                    })
                                    continue

                                row = {
                                    "index": i,
                                    "text": text,
                                    "predicted_class": resp.predicted_class,
                                    "predicted_class_idx": (
                                        resp.predicted_class_idx
                                    ),
                                    "confidence": resp.confidence,
                                    "timestamp": resp.timestamp.isoformat(),
                                }
                                # Add probabilities as separate columns
                                for label, prob in resp.probabilities.items():
                                    row[f"prob_{label}"] = prob

                                # Add explanation if present
                                if batch_explain and resp.explanation:
                                    row["explanation_method"] = (
                                        resp.explanation.method
                                    )
                                    row["top_contributors"] = "; ".join(
                                        [
                                            f"{w}({c:+.2f})"
                                            for w, c
                                            in resp.explanation.top_contributors[:batch_n_explain]  # noqa: E501
                                        ]
                                    )

                                output_rows.append(row)

                            output_df = pd.DataFrame(output_rows)

                            # Display summary
                            st.subheader("📊 Results Summary")
                            col_sum1, col_sum2, col_sum3 = st.columns(3)
                            with col_sum1:
                                st.metric("Total Processed", len(output_df))
                            with col_sum2:
                                if "error" in output_df.columns:
                                    success_count = len(
                                        output_df[output_df["error"].isna()]
                                    )
                                else:
                                    success_count = len(output_df)
                                st.metric("Successful", success_count)
                            with col_sum3:
                                if "predicted_class" in output_df.columns:
                                    top_class = output_df[
                                        "predicted_class"
                                    ].mode()[0]
                                    st.metric("Most Common", top_class)

                            # Download buttons
                            csv_data = output_df.to_csv(
                                index=False, encoding="utf-8"
                            )
                            timestamp = datetime.now().strftime(
                                "%Y%m%d_%H%M%S"
                            )
                            st.download_button(
                                "📥 Download results (CSV)",
                                data=csv_data,
                                file_name=f"predictions_{timestamp}.csv",
                                mime="text/csv",
                                type="primary",
                            )

                            json_data = output_df.to_json(
                                orient='records', force_ascii=False, indent=2
                            )
                            st.download_button(
                                "📥 Download results (JSON)",
                                data=json_data,
                                file_name=f"predictions_{timestamp}.json",
                                mime="application/json",
                            )

                            # Preview results
                            with st.expander("👀 Preview results"):
                                st.dataframe(
                                    output_df.head(), width='stretch'
                                )

                            logger.info(
                                "Dashboard batch prediction: "
                                f"{len(texts)} texts processed"
                            )

                        except Exception as e:
                            st.error(
                                "❌ Batch prediction failed: "
                                f"{type(e).__name__}: {e}"
                            )
                            logger.error(
                                f"Dashboard batch error: {e}",
                                exc_info=True
                            )

        except Exception as e:
            st.error(f"❌ Failed to read CSV: {e}")
            logger.error(f"Dashboard CSV read error: {e}")

# ---------------------------------------------------------------------------
# Tab 3: About / Info
# ---------------------------------------------------------------------------

with tab_info:
    st.header("ℹ️ About this Dashboard")

    import requests

    st.markdown(f"""
    ### {API_TITLE}

    **Version:** {API_VERSION}

    This interactive dashboard allows you to:

    1. 🔍 **Predict sentiment** for individual airline tweets
    2. 📦 **Batch process** CSV files with multiple tweets
    3. 🔍 **Explore explanations** — see which words influenced the prediction
    4. 📊 **Visualize probabilities** as interactive bar charts

    ---

    ### How it works

    - Model: Logistic Regression with TF-IDF vectorization
    - Training: Confidence-aware learning from airline tweet dataset
    - Interpretation: Model weights (+ optional SHAP values)

    ---

    ### Technical Details

    | Component | Value |
    |-----------|-------|
    | Framework | Streamlit {st.__version__} |
    | Model Path | `{MODEL_PATH}` |
    | SHAP Support | {"✅ Yes" if SHAP_AVAILABLE else "❌ No"} |
    | Max Text Length | 1000 characters |
    | Max Batch Size | 100 texts |

    ---

    ### Links

    - 📚 [Swagger UI Documentation](
                http://localhost:8000/docs) — Interactive API docs
    - 📚 [ReDoc API Documentation](
                http://localhost:8000/redoc) — Alternative view
    """)

    st.info(
        "💡 **API Required:**\n\n"
        "Interactive documentation requires the backend API "
        "to be running on `http://localhost:8000`.\n\n"
        "Open a new terminal and execute:\n"
        "```\nuvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload\n```"  # noqa: E501
    )

    try:
        resp = requests.get("http://localhost:8000/health", timeout=1)
        api_available = resp.status_code == 200
    except requests.RequestException:
        api_available = False

    if api_available:
        st.success("🔌 API server detected — documentation links are active")
    else:
        st.caption(
            "💡 API server not detected on port 8000 — "
            "documentation links will not work"
        )

    st.markdown("""
    - 🔧 [GitHub Repository](
    https://github.com/IlyaShaposhnikov/airline-sentiment-analysis
    ) — Source code
    - 📊 [Kaggle Dataset](
    https://www.kaggle.com/crowdflower/twitter-airline-sentiment
    ) — Training data
    """)

    # Footer
    st.divider()
    st.caption(
        "Built with ❤️ using Streamlit • "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d')}"
    )

# ============================================================================
# Footer (global)
# ============================================================================

st.divider()
col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    st.caption("💡 Tip: Use the sidebar to reload the model or check status")
with col_f2:
    if st.button("⬆️ Scroll to Top"):
        st.rerun()  # Simple way to "scroll" by refreshing
