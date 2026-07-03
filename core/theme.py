"""
Shared visual theme: CSS injection + reusable header/badge components used
across every page, so the app has a consistent, polished look instead of
Streamlit's plain defaults.
"""

import streamlit as st

_CSS = """
<style>
.auditor-header {
    padding: 1.75rem 2rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #1f2a44 0%, #3b3f8f 55%, #6a3f9e 100%);
    color: white;
    margin-bottom: 1.5rem;
}
.auditor-header h1 {
    margin: 0;
    font-size: 1.9rem;
    color: white;
}
.auditor-header p {
    margin: 0.35rem 0 0 0;
    opacity: 0.88;
    font-size: 1rem;
    color: white;
}
div[data-testid="stMetric"] {
    background: rgba(120, 120, 160, 0.08);
    border: 1px solid rgba(120, 120, 160, 0.18);
    border-radius: 12px;
    padding: 0.85rem 1rem 0.65rem 1rem;
}
.status-badge {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.status-badge.llm {
    background: rgba(46, 204, 113, 0.18);
    color: #1e8449;
}
.status-badge.local {
    background: rgba(241, 196, 15, 0.20);
    color: #9a7d0a;
}
.feature-card {
    border: 1px solid rgba(120, 120, 160, 0.18);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    height: 100%;
    background: rgba(120, 120, 160, 0.05);
}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="auditor-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(agent_mode: str):
    if agent_mode == "llm":
        st.sidebar.markdown('<span class="status-badge llm">🟢 LLM Mode</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="status-badge local">🟡 Local Mode</span>', unsafe_allow_html=True)
