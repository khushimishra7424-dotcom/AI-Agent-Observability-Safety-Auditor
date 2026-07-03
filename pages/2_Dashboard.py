import json

import pandas as pd
import plotly.express as px
import streamlit as st

from core.database import init_db, fetch_all_queries
from core.agent import TargetAgent
from core.theme import inject_css, render_header, render_status_badge

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
inject_css()


@st.cache_resource
def load_agent():
    return TargetAgent(kb_dir="data/knowledge_base")


init_db()
agent = load_agent()
render_status_badge(agent.mode)

render_header("📊 Observability Dashboard", "Historical view of every query the target agent has answered, and how it was audited.")

rows = fetch_all_queries()

if not rows:
    st.warning("No audit history yet. Go to **Live Audit** and ask a few questions first.")
    st.stop()

df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["sources"] = df["sources"].apply(lambda s: ", ".join(json.loads(s)) if s else "none")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Queries Audited", len(df))
k2.metric("Avg Trust Score", f"{df['trust_score'].mean():.1f}/100")
flagged_pct = 100 * df["hallucination_flag"].mean()
k3.metric("Hallucination Flag Rate", f"{flagged_pct:.1f}%")
k4.metric("Avg Confidence", f"{100 * df['confidence'].mean():.1f}%")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Trust Score Over Time")
    fig = px.line(df.sort_values("timestamp"), x="timestamp", y="trust_score", markers=True)
    fig.add_hline(y=75, line_dash="dot", line_color="green", annotation_text="High trust threshold")
    fig.add_hline(y=45, line_dash="dot", line_color="orange", annotation_text="Low trust threshold")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Hallucination Score Distribution")
    fig2 = px.histogram(df, x="hallucination_score", nbins=10, color="hallucination_flag",
                         color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                         labels={"hallucination_flag": "Flagged"})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("🚩 Flagged Responses (needs human review)")
flagged_df = df[df["hallucination_flag"] == 1][
    ["timestamp", "query", "answer", "hallucination_score", "trust_score", "sources"]
]
if flagged_df.empty:
    st.success("No flagged responses yet. Nice.")
else:
    st.dataframe(flagged_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("📋 Full Query Log")
st.dataframe(
    df[["timestamp", "query", "answer", "confidence", "hallucination_score", "trust_score", "sources", "user_feedback"]],
    use_container_width=True,
    hide_index=True,
)
