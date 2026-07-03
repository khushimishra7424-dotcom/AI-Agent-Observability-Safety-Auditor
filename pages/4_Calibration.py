import plotly.graph_objects as go
import streamlit as st

from core.database import init_db, fetch_calibration_pairs
from core.agent import TargetAgent
from core.theme import inject_css, render_header, render_status_badge
from auditor.calibration import compute_calibration

st.set_page_config(page_title="Calibration", page_icon="📈", layout="wide")
inject_css()


@st.cache_resource
def load_agent():
    return TargetAgent(kb_dir="data/knowledge_base")


init_db()
agent = load_agent()
render_status_badge(agent.mode)

render_header(
    "📈 Confidence Calibration",
    "A well-calibrated agent's stated confidence should match its real-world accuracy. "
    "This page compares them using feedback you gave on the Live Audit page.",
)

pairs = fetch_calibration_pairs()

if not pairs:
    st.warning(
        "No labeled feedback yet. Go to **Live Audit**, ask a few questions, and click "
        "👍 Correct / 👎 Incorrect on the answers to build up calibration data."
    )
    st.stop()

result = compute_calibration(pairs, n_bins=5)

c1, c2 = st.columns(2)
c1.metric("Expected Calibration Error (ECE)", f"{result.ece:.3f}", help="Lower is better. 0 = perfectly calibrated.")
c2.metric("Labeled Samples", result.n_samples)

fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                          line=dict(dash="dash", color="gray")))
fig.add_trace(go.Scatter(
    x=result.bin_confidence, y=result.bin_accuracy, mode="markers+lines",
    name="Agent (observed)",
    marker=dict(size=[max(8, c * 3) for c in result.bin_counts], color="crimson"),
))
fig.update_layout(
    xaxis_title="Predicted Confidence", yaxis_title="Observed Accuracy",
    xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
    title="Reliability Diagram",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**How to read this chart:** the dashed diagonal is what a *perfectly*
calibrated agent looks like. Points **below** the diagonal mean the agent is
**overconfident**; points **above** mean it's **underconfident**. Marker size
shows how many samples fall in that bin.
""")

with st.expander("📋 Raw bin data"):
    st.write({
        "bin_edges": [round(e, 2) for e in result.bin_edges],
        "bin_confidence": result.bin_confidence,
        "bin_accuracy": result.bin_accuracy,
        "bin_counts": result.bin_counts,
    })
