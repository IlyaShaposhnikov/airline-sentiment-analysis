#!/usr/bin/env python3
"""
Streamlit dashboard for interactive sentiment analysis.

Provides a user-friendly web interface for:
- Choosing any trained model bundle found in artifacts/ (or, when there is
  none, in the committed demo set demo/models/ used by the hosted demo)
- Single text prediction with confidence visualization
- Optional explanation toggle (top contributing words)
- Batch prediction via CSV upload/download
- Side-by-side comparison of several models on the same text

Usage:
    streamlit run src/dashboard.py
"""
from datetime import datetime
from pathlib import Path
import sys
import time

import pandas as pd
import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.config import (  # noqa: E402
    API_TITLE,
    API_VERSION,
    MAX_BATCH_SIZE,
    MAX_EXPLAIN_COUNT,
    MAX_TEXT_LENGTH,
    get_model_path,
)
from src.api.services import ModelService  # noqa: E402
from src.interpretability import SHAP_AVAILABLE  # noqa: E402
from src.model_registry import (  # noqa: E402
    BundleInfo,
    DEMO_MODELS_DIRNAME,
    choose_models_dir,
    default_bundle_index,
    discover_bundles,
)
from src.utils.logging_config import setup_logger  # noqa: E402

# Configure logging (Streamlit captures stdout/stderr)
logger = setup_logger("dashboard", level="INFO")

MODELS_DIR = choose_models_dir(PROJECT_ROOT)
IS_DEMO = MODELS_DIR == PROJECT_ROOT / DEMO_MODELS_DIRNAME
try:
    MODELS_DIR_LABEL = MODELS_DIR.relative_to(PROJECT_ROOT).as_posix()
except ValueError:
    MODELS_DIR_LABEL = str(MODELS_DIR)

# ============================================================================
# Initialize session state for explanation settings (persist across reloads)
# ============================================================================

# Session state keys persist user settings across Streamlit reruns
for key, default in {
    "single_explain": False,
    "single_shap": False,
    "single_n_explain": 5,
    "batch_explain": False,
    "batch_shap": False,
    "batch_n_explain": 5,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title=API_TITLE,
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(f"✈️ {API_TITLE}")
st.caption(
    f"Version {API_VERSION} — Interactive sentiment analysis "
    "for airline tweets"
)

# ============================================================================
# Model loading (cached per bundle; shared sentence-embedding encoders)
# ============================================================================


@st.cache_resource(show_spinner=False)
def shared_encoders() -> dict:
    """One loaded encoder per embedding model, shared by all bundles."""
    return {}


@st.cache_resource(show_spinner="Loading model...")
def get_service(bundle_path: str) -> ModelService:
    """
    Load a bundle once per dashboard process (cached by path).

    Failures are not cached: the next rerun tries again.
    """
    service = ModelService(
        model_path=bundle_path, encoder_cache=shared_encoders()
    )
    service.load()
    return service


def load_hint(info: BundleInfo) -> str:
    """Actionable hint for a bundle that failed to load."""
    if info.needs_deep_learning:
        return (
            "This model needs the optional deep-learning dependencies: "
            "`pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu` and "
            "`pip install -r requirements-dl.txt`."
        )
    return "Retrain it with `python scripts/train.py`."


bundles = discover_bundles(MODELS_DIR)
if not bundles:
    st.error(f"❌ No trained models found in `{MODELS_DIR_LABEL}/`")
    st.info(
        "💡 Run `python scripts/train.py` first to train and save a model"
    )
    st.stop()

bundle_by_path = {str(b.path): b for b in bundles}
bundle_paths = list(bundle_by_path)


def bundle_label(path: str) -> str:
    return bundle_by_path[path].label


# ============================================================================
# Sidebar: model selection & info
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Model")
    # Default: the bundle the API serves (MODEL_PATH)
    if st.session_state.get("model_path") not in bundle_by_path:
        st.session_state.model_path = bundle_paths[
            default_bundle_index(bundles, get_model_path())
        ]
    selected_path = st.selectbox(
        "Model bundle",
        bundle_paths,
        format_func=bundle_label,
        key="model_path",
        help=(
            f"Trained bundles found in {MODELS_DIR_LABEL}/. F1 is the macro "
            "F1 on the test split of that run (binary runs exclude neutral "
            "tweets)."
        ),
    )
    selected = bundle_by_path[selected_path]
    if IS_DEMO:
        st.caption(
            "🌐 Demo models (`demo/models/`), trained offline with "
            "`scripts/train.py`. Embedding models download a 90 MB encoder "
            "on first use."
        )

try:
    service = get_service(selected_path)
    load_error = None
except Exception as e:  # missing optional deps, corrupt file, ...
    service = None
    load_error = f"{type(e).__name__}: {e}"
    logger.error(f"Dashboard failed to load {selected_path}: {load_error}")

MODEL_READY = service is not None

# Word-level explanations need TF-IDF features and a linear model;
# embedding / MLP bundles predict normally but cannot explain
EXPLAIN_OK = bool(service and service.explanations_supported)
EXPLAIN_HELP = (
    None if EXPLAIN_OK else
    "Not available for this model: word-level explanations require "
    "TF-IDF features and a linear model (logistic regression)."
)

# Visual status banner
if MODEL_READY:
    st.success(f"✅ Model loaded: {selected.label}")
else:
    st.error(f"❌ Could not load `{selected.name}`: {load_error}")
    st.info(f"💡 {load_hint(selected)}")

with st.sidebar:
    try:
        shown_path = selected.path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown_path = selected.path
    st.code(f"Path: {shown_path.as_posix()}", language="text")
    if MODEL_READY:
        info = service.model_info
        st.caption(
            f"Classes: **{', '.join(service._class_names)}**  \n"
            f"Classifier: `{info['model_type']}`  \n"
            f"Features: `{info['vectorizer_type']}`"
        )
        if not EXPLAIN_OK:
            st.caption("ℹ️ Explanations are not available for this model")
    if selected.accuracy is not None:
        st.metric("Test accuracy", f"{selected.accuracy:.3f}")

    shap_status = "Installed" if SHAP_AVAILABLE else "Not installed"
    st.metric("SHAP Library", shap_status)
    if SHAP_AVAILABLE:
        st.caption(
            "ℹ️ SHAP explanations may use zero baseline without training data"
        )

    st.divider()

    st.subheader("Quick Tips")
    st.markdown("""
    - 🧠 Pick a model above; each one is loaded once and cached
    - ✍️ Enter a tweet to see sentiment prediction
    - 🔍 Toggle "Explain" to see top contributing words (TF-IDF + LogReg)
    - 📁 Upload CSV for batch prediction
    - ⚖️ Compare all models on the same text in the last tab
    """)

    st.divider()

    # Reload: forget loaded models (e.g. after retraining) and rescan
    if st.button("🔄 Reload models"):
        get_service.clear()
        shared_encoders.clear()
        st.rerun()

# ============================================================================
# Tabbed interface: Single / Batch / Compare / Info
# ============================================================================

tab_single, tab_batch, tab_compare, tab_info = st.tabs([
    "🔍 Single Prediction",
    "📦 Batch Prediction",
    "⚖️ Compare Models",
    "ℹ️ About",
])

# ---------------------------------------------------------------------------
# Tab 1: Single text prediction
# ---------------------------------------------------------------------------

with tab_single:
    st.header("Predict sentiment for a single tweet")

    # Explanation settings: persisted via session_state keys
    col1, col2 = st.columns([1, 2])
    with col1:
        explain_toggle = st.checkbox(
            "🔍 Show explanation",
            disabled=not EXPLAIN_OK,
            help=EXPLAIN_HELP,
            key="single_explain"
        ) and EXPLAIN_OK
    with col2:
        use_shap_toggle = st.checkbox(
            "Use SHAP (if available)",
            disabled=not (SHAP_AVAILABLE and EXPLAIN_OK),
            key="single_shap"
        )
        if not SHAP_AVAILABLE and use_shap_toggle:
            st.caption(
                "ℹ️ SHAP not installed — will fall back to model weights"
            )

    n_explain = st.slider(
        "Number of top words to show",
        min_value=1,
        max_value=MAX_EXPLAIN_COUNT,
        disabled=not explain_toggle,
        key="single_n_explain"
    )

    # Form ensures single submission per click (prevents duplicate predictions)
    with st.form("single_prediction_form"):
        text_input = st.text_area(
            "Enter tweet text",
            height=100,
            placeholder="@VirginAmerica Great flight, excellent service!",
            max_chars=MAX_TEXT_LENGTH,
            key="single_text_input"
        )

        submitted = st.form_submit_button(
            "🚀 Predict", type="primary", disabled=not MODEL_READY
        )

    # Process prediction only on form submit with non-empty text
    if submitted and text_input.strip():
        with st.spinner("Analyzing sentiment..."):
            try:
                response = service.predict_single_sync(
                    text=text_input.strip(),
                    explain=explain_toggle,
                    use_shap=st.session_state.single_shap and SHAP_AVAILABLE,
                    n_explain=st.session_state.single_n_explain,
                )

                st.subheader("📊 Result")

                # Prediction header: sentiment badge + confidence
                col_prob, col_class = st.columns([2, 1])
                with col_class:
                    # Color-coded badge for quick visual feedback
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
                    # Horizontal bar chart for probability distribution
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

                # Explanation section: top contributing words
                # with color-coded contributions
                if explain_toggle and response.explanation:
                    st.subheader("🔍 Explanation")
                    st.caption(f"Method: `{response.explanation.method}`")

                    contrib_df = pd.DataFrame(
                        response.explanation.top_contributors,
                        columns=["Word", "Contribution"]
                    )
                    contrib_df["Contribution"] = contrib_df[
                        "Contribution"
                    ].apply(lambda x: f"{x:+.3f}")

                    # Color-code contributions:
                    # green=positive, red=negative influence
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

                # Raw JSON toggle for developers/debugging
                with st.expander("🔧 View raw response (JSON)"):
                    st.json(response.model_dump(mode='json'), expanded=False)

                logger.info(
                    f"Dashboard prediction [{selected.name}]: "
                    f"'{text_input[:50]}...' → {response.predicted_class}"
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

    st.markdown(f"""
    **Upload a CSV file** with a `text` column containing tweets
    (up to {MAX_BATCH_SIZE} rows). Uses the model selected in the sidebar.

    Example format:
    ```csv
    text
    "Great flight, thanks!"
    "Terrible delay, never again"
    "Okay experience, nothing special"
    ```
    """)

    uploaded_file = st.file_uploader(
        "📁 Choose CSV file", type=["csv"], disabled=not MODEL_READY
    )

    if uploaded_file:
        # Read and validate uploaded CSV
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

                with st.expander("👀 Preview uploaded data"):
                    st.dataframe(df.head(), width='stretch')

                # Batch explanation settings (persisted via session_state)
                col1, col2 = st.columns(2)
                with col1:
                    batch_explain = st.checkbox(
                        "🔍 Include explanations",
                        disabled=not EXPLAIN_OK,
                        help=EXPLAIN_HELP,
                        key="batch_explain"
                    ) and EXPLAIN_OK
                with col2:
                    st.checkbox(
                        "Use SHAP (if available)",
                        disabled=not (SHAP_AVAILABLE and EXPLAIN_OK),
                        key="batch_shap"
                    )

                batch_n_explain = st.slider(
                    "Top words per explanation",
                    min_value=1,
                    max_value=MAX_EXPLAIN_COUNT,
                    disabled=not batch_explain,
                    key="batch_n_explain"
                )

                with st.form("batch_prediction_form"):
                    batch_submitted = st.form_submit_button(
                        "🚀 Run Batch Prediction", type="primary"
                    )

                if batch_submitted:
                    with st.spinner(f"Processing {len(df)} texts..."):
                        try:
                            texts = df["text"].dropna().astype(str).tolist()
                            results = []

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for i, text in enumerate(texts):
                                try:
                                    response = service.predict_single_sync(
                                        text=text,
                                        explain=batch_explain,
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
                                    # Log failure but continue processing
                                    # (partial success)
                                    logger.warning(
                                        f"Batch item {i} failed: {e}"
                                    )
                                    results.append(None)

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

                            # Flatten results for CSV/JSON export
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
                                    "model": selected.name,
                                    "predicted_class": resp.predicted_class,
                                    "predicted_class_idx": (
                                        resp.predicted_class_idx
                                    ),
                                    "confidence": resp.confidence,
                                    "timestamp": resp.timestamp.isoformat(),
                                }
                                for label, prob in resp.probabilities.items():
                                    row[f"prob_{label}"] = prob

                                if batch_explain and resp.explanation:
                                    row["explanation_method"] = (
                                        resp.explanation.method
                                    )
                                    contributors = (
                                        resp.explanation.top_contributors
                                    )[:batch_n_explain]
                                    row["top_contributors"] = "; ".join(
                                        f"{w}({c:+.2f})"
                                        for w, c in contributors
                                    )

                                output_rows.append(row)

                            output_df = pd.DataFrame(output_rows)

                            st.subheader("📊 Results Summary")
                            col_sum1, col_sum2, col_sum3 = st.columns(3)
                            with col_sum1:
                                st.metric("Total Processed", len(output_df))
                            with col_sum2:
                                if "error" in output_df.columns:
                                    success_count = int(
                                        output_df["error"].isna().sum()
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

                            with st.expander("👀 Preview results"):
                                st.dataframe(
                                    output_df.head(), width='stretch'
                                )

                            logger.info(
                                f"Dashboard batch prediction [{selected.name}]"
                                f": {len(texts)} texts processed"
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
# Tab 3: Compare models on one text
# ---------------------------------------------------------------------------

with tab_compare:
    st.header("Compare models on the same text")
    st.caption(
        "Every selected model predicts the same text. Models are loaded once "
        "and cached, so the latency column is the cost of one prediction "
        "(CPU, end-to-end from raw text). Binary models have no neutral "
        "class."
    )

    compare_paths = st.multiselect(
        "Models",
        bundle_paths,
        default=bundle_paths,
        format_func=bundle_label,
        key="compare_models",
    )

    with st.form("compare_form"):
        compare_text = st.text_area(
            "Enter tweet text",
            height=100,
            placeholder="@united Delayed again, but the crew was lovely",
            max_chars=MAX_TEXT_LENGTH,
            key="compare_text_input",
        )
        compare_submitted = st.form_submit_button(
            "⚖️ Compare", type="primary", disabled=not compare_paths
        )

    if compare_submitted and compare_text.strip():
        rows = []
        progress = st.progress(0)
        for i, path in enumerate(compare_paths):
            info = bundle_by_path[path]
            row = {"Model": info.name, "Setup": info.description or "—"}
            try:
                with st.spinner(f"Loading {info.name}..."):
                    model_service = get_service(path)
                start = time.perf_counter()
                response = model_service.predict_single_sync(
                    text=compare_text.strip()
                )
                row["Latency, ms"] = (time.perf_counter() - start) * 1000
                row["Prediction"] = response.predicted_class
                row["Confidence"] = response.confidence
                for label, prob in response.probabilities.items():
                    row[f"P({label})"] = prob
            except Exception as e:
                row["Error"] = f"{type(e).__name__}: {e}"
                logger.warning(f"Compare: {info.name} failed: {e}")
            rows.append(row)
            progress.progress((i + 1) / len(compare_paths))
        progress.empty()

        result_df = pd.DataFrame(rows)
        ordered = ["Model", "Setup", "Prediction", "Confidence"] + [
            c for c in result_df.columns if c.startswith("P(")
        ] + ["Latency, ms", "Error"]
        result_df = result_df[[c for c in ordered if c in result_df.columns]]

        st.dataframe(
            result_df,
            hide_index=True,
            width='stretch',
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "Latency, ms": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        if "Error" in result_df.columns:
            n_failed = int(result_df["Error"].notna().sum())
            st.warning(
                f"⚠️ {n_failed} model(s) could not run — see the Error "
                "column (embedding / MLP models need requirements-dl.txt)"
            )

        if "Prediction" in result_df.columns:
            votes = result_df["Prediction"].dropna().value_counts()
            if len(votes) == 1:
                st.success(f"All models agree: **{votes.index[0]}**")
            elif len(votes) > 1:
                summary = ", ".join(f"{k}: {v}" for k, v in votes.items())
                st.info(f"Models disagree — {summary}")

        logger.info(
            f"Dashboard comparison: {len(compare_paths)} models, "
            f"'{compare_text[:50]}...'"
        )

    elif compare_submitted:
        st.warning("⚠️ Please enter some text to analyze")

# ---------------------------------------------------------------------------
# Tab 4: About / Info
# ---------------------------------------------------------------------------

with tab_info:
    st.header("ℹ️ About this Dashboard")

    import requests

    st.markdown(f"""
    ### {API_TITLE}

    **Version:** {API_VERSION}

    This interactive dashboard allows you to:

    1. 🧠 **Choose a model** — any bundle in `{MODELS_DIR_LABEL}/`
    2. 🔍 **Predict sentiment** for individual airline tweets
    3. 📦 **Batch process** CSV files with multiple tweets
    4. 🔍 **Explore explanations** — which words influenced the prediction
    5. ⚖️ **Compare models** side by side on the same text

    ---

    ### How it works

    - Features: TF-IDF or sentence embeddings (Sentence-Transformers)
    - Classifiers: logistic regression or a PyTorch MLP
    - Training: confidence-aware learning from the airline tweet dataset
      (`python scripts/train.py`, see README for the measured comparison)
    - Interpretation: model weights (+ optional SHAP values) for
      TF-IDF + logistic regression
    - Selected model: {selected.label}

    ---

    ### Technical Details

    | Component | Value |
    |-----------|-------|
    | Framework | Streamlit {st.__version__} |
    | Models found | {len(bundles)} in `{MODELS_DIR_LABEL}/` |
    | SHAP Support | {"✅ Yes" if SHAP_AVAILABLE else "❌ No"} |
    | Max Text Length | {MAX_TEXT_LENGTH} characters |
    | Max Batch Size | {MAX_BATCH_SIZE} texts |

    ---

    ### Links

    - 📚 [Swagger UI Documentation](
                http://localhost:8000/docs) — Interactive API docs
    - 📚 [ReDoc API Documentation](
                http://localhost:8000/redoc) — Alternative view
    """)

    # Check if backend API is running (for documentation links)
    st.info(
        "💡 **API Required:**\n\n"
        "Interactive documentation requires the backend API "
        "to be running on `http://localhost:8000`. The API serves one "
        "model (`MODEL_PATH`); the dashboard can switch between all of "
        "them.\n\n"
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

    st.divider()
    st.caption(
        "Built with ❤️ using Streamlit • "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d')}"
    )

# ============================================================================
# Footer (global)
# ============================================================================

st.divider()
st.caption("💡 Tip: Use the sidebar to switch or reload models")
