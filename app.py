import pandas as pd
import streamlit as st

from core.agent import TargetAgent
from core.database import init_db, fetch_all_queries
from core.theme import inject_css, render_header, render_status_badge

st.set_page_config(
    page_title="AI Agent Observability & Safety Auditor",
    page_icon="🛡️",
    layout="wide",
)

inject_css()


@st.cache_resource
def load_agent():
    return TargetAgent(kb_dir="data/knowledge_base")


init_db()
agent = load_agent()
render_status_badge(agent.mode)

render_header(
    "🛡️ AI Agent Observability & Safety Auditor",
    "An AI system that monitors, evaluates, and red-teams another AI system in real time.",
)

st.markdown("""
### What is this project?

Most AI projects build *an* AI system. This one builds the layer that sits
**on top of** an AI system to answer the question every company deploying
agents eventually asks: *"How do we know when our AI is wrong, biased, or
being manipulated?"*

This dashboard audits a small RAG-based policy assistant (the **Target
Agent**, built on a company HR / product / loan-policy knowledge base) using
four independent auditor modules:
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Hallucination Detector")
    st.write("Flags answers that aren't grounded in the retrieved source documents, using real sentence embeddings plus an optional LLM judge.")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("#### ⚖️ Bias / Fairness Checker")
    st.write("Runs paired counterfactual prompts to test answer consistency across demographics.")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("#### 🛡️ Prompt Injection Red-Team")
    st.write("Fires a suite of adversarial prompts to test if the agent can be hijacked.")
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("#### 📈 Confidence Calibration")
    st.write("Tracks whether the agent's confidence actually matches its real-world accuracy.")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

rows = fetch_all_queries()
if rows:
    st.markdown("### 📊 Quick stats")
    df = pd.DataFrame(rows)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Queries Audited", len(df))
    k2.metric("Avg Trust Score", f"{df['trust_score'].mean():.1f}/100")
    k3.metric("Hallucination Flag Rate", f"{100 * df['hallucination_flag'].mean():.1f}%")
    k4.metric("Avg Confidence", f"{100 * df['confidence'].mean():.1f}%")
    st.divider()

st.markdown("""
### 👉 Use the sidebar to navigate:
- **Live Audit** — ask the agent a question and see the full audit breakdown in real time
- **Dashboard** — historical trust score trends and flagged response analytics
- **Red Team Test** — run the full adversarial prompt injection suite on demand
- **Calibration** — see the reliability curve and Expected Calibration Error (ECE)

### 📚 Knowledge base currently loaded
The target agent can answer questions about:
- Employee leave policy (`hr_policy.txt`)
- CloudSync product FAQ (`product_faq.txt`)
- Personal loan eligibility policy (`loan_policy.txt`)

Try asking something **in-domain** (e.g. *"How many leave days do I get?"*) vs
something **out-of-domain** (e.g. *"What is the capital of France?"*) on the
Live Audit page to see the Hallucination Detector in action.
""")
