"""
Entry point for the hosted demo (Streamlit Community Cloud).

Runs the regular dashboard. Kept in demo/ so that Streamlit Community Cloud
installs demo/requirements.txt (lean, CPU-only torch) instead of the
development requirements in the project root. Models come from
demo/models/ (see scripts/export_demo.py).

Local equivalent: streamlit run src/dashboard.py
"""
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "dashboard.py"

# Executed on every Streamlit rerun, exactly like a main script
exec(
    compile(DASHBOARD.read_text(encoding="utf-8"), str(DASHBOARD), "exec"),
    {"__name__": "__main__", "__file__": str(DASHBOARD)},
)
