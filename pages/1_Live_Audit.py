import streamlit as st

from core.agent import TargetAgent
from core.database import init_db, log_query, update_feedback
from core.theme import inject_css, render_header, render_status_badge
from auditor.hallucination import HallucinationDetector
from auditor.trust_score import compute_trust_score

st.set_page_config(page_title="Live Audit", page_icon="🔍", layout="wide")
inject_css()


@st.cache_resource
def load_agent():
    return TargetAgent(kb_dir="data/knowledge_base")


init_db()
agent = load_agent()
render_status_badge(agent.mode)
detector = HallucinationDetector(
    store=agent.store,
    llm_client=agent._client if agent.mode == "llm" else None,
)

render_header("🔍 Live Audit", "Ask the target agent a question and see the full real-time audit breakdown.")

col_a, col_b = st.columns([3, 1])
with col_a:
    query = st.text_input("Ask a question", placeholder="e.g. How many leave days do I get per year?")
with col_b:
    force_no_context = st.checkbox("Force no context (simulate hallucination)", value=False)

if st.button("Run Audit", type="primary") and query.strip():
    with st.spinner("Running agent + auditors..."):
        response = agent.generate(query, force_no_context=force_no_context)
        report = detector.check(response.answer, response.retrieved)
        trust = compute_trust_score(confidence=response.confidence, hallucination_score=report.hallucination_score)

        sources = [r.chunk.source for r in response.retrieved]
        row_id = log_query(
            query=query,
            answer=response.answer,
            confidence=response.confidence,
            hallucination_score=report.hallucination_score,
            hallucination_flag=report.is_flagged,
            trust_score=trust.trust_score,
            sources=sources,
        )
        st.session_state["last_row_id"] = row_id
        st.session_state["last_response"] = response
        st.session_state["last_report"] = report
        st.session_state["last_trust"] = trust

if "last_response" in st.session_state:
    response = st.session_state["last_response"]
    report = st.session_state["last_report"]
    trust = st.session_state["last_trust"]

    st.divider()
    st.subheader("🤖 Agent Answer")
    st.write(response.answer)

    has_llm_judge = report.llm_judge_score is not None
    metric_cols = st.columns(5 if has_llm_judge else 4)
    metric_cols[0].metric("Trust Score", f"{trust.trust_score}/100", trust.label)
    metric_cols[1].metric("Agent Confidence", f"{response.confidence * 100:.0f}%")
    metric_cols[2].metric(
        "Hallucination Score", f"{report.hallucination_score * 100:.0f}%",
        delta="FLAGGED" if report.is_flagged else "OK",
        delta_color="inverse" if report.is_flagged else "normal",
    )
    metric_cols[3].metric("Retrieval Match", f"{report.retrieval_strength * 100:.0f}%")
    if has_llm_judge:
        metric_cols[4].metric("LLM Judge Score", f"{report.llm_judge_score * 100:.0f}%")

    if report.is_flagged:
        st.error(f"⚠️ {report.explanation}")
    else:
        st.success(f"✅ {report.explanation}")

    with st.expander("📄 Retrieved source chunks used for this answer"):
        if response.retrieved:
            for r in response.retrieved:
                st.markdown(f"**{r.chunk.source}** (relevance: {r.score:.2f})")
                st.caption(r.chunk.text)
        else:
            st.write("No documents were retrieved (or context was force-disabled).")

    st.divider()
    st.markdown("**Was this answer actually correct?** (used to build the calibration curve)")
    fb1, fb2 = st.columns(2)
    if fb1.button("👍 Correct"):
        update_feedback(st.session_state["last_row_id"], "correct")
        st.toast("Feedback logged: correct")
    if fb2.button("👎 Incorrect"):
        update_feedback(st.session_state["last_row_id"], "incorrect")
        st.toast("Feedback logged: incorrect")
