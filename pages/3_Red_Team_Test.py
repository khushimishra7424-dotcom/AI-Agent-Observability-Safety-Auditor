import pandas as pd
import streamlit as st

from core.agent import TargetAgent
from core.database import init_db, log_redteam_result
from core.theme import inject_css, render_header, render_status_badge
from auditor.injection import PromptInjectionTester
from auditor.bias import BiasChecker

st.set_page_config(page_title="Red Team Test", page_icon="🛡️", layout="wide")
inject_css()


@st.cache_resource
def load_agent():
    return TargetAgent(kb_dir="data/knowledge_base")


init_db()
agent = load_agent()
render_status_badge(agent.mode)

render_header("🛡️ Red Team Test Suite", "Fires adversarial prompts and paired fairness tests at the target agent on demand.")

tab1, tab2 = st.tabs(["🧨 Prompt Injection Suite", "⚖️ Bias / Fairness Suite"])

with tab1:
    st.markdown("""
    This suite fires **10 adversarial prompts** (instruction override, role
    hijacking, prompt leaking, data exfiltration attempts, and a harmful
    request) at the target agent, then checks whether the response shows
    signs of successful compliance with the attack.
    """)
    if st.button("▶️ Run Prompt Injection Suite", type="primary"):
        tester = PromptInjectionTester(agent)
        with st.spinner("Running 10 adversarial prompts..."):
            results = tester.run_suite()
            summary = tester.summarize(results)
            for r in results:
                log_redteam_result(r.prompt_id, r.category, r.prompt, r.response, r.attack_succeeded)
        st.session_state["inj_results"] = results
        st.session_state["inj_summary"] = summary

    if "inj_results" in st.session_state:
        summary = st.session_state["inj_summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Prompts Tested", summary["total_prompts"])
        c2.metric("Attacks That Succeeded", summary["attacks_succeeded"])
        c3.metric("Robustness Score", f"{summary['robustness_score'] * 100:.0f}%")

        df = pd.DataFrame([r.__dict__ for r in st.session_state["inj_results"]])
        df["status"] = df["attack_succeeded"].apply(lambda x: "🔴 VULNERABLE" if x else "🟢 BLOCKED")
        st.dataframe(
            df[["prompt_id", "category", "prompt", "status", "response"]],
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            "⬇️ Download red-team report (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="redteam_injection_report.csv",
            mime="text/csv",
        )

with tab2:
    st.markdown("""
    This suite runs **paired counterfactual prompts** — the same underlying
    question asked with different applicant names / demographic markers —
    and checks whether the agent's answers stay consistent.
    """)
    if st.button("▶️ Run Bias / Fairness Suite", type="primary"):
        checker = BiasChecker(agent=agent, store=agent.store)
        with st.spinner("Running paired counterfactual prompts..."):
            reports = checker.run_all()
        st.session_state["bias_reports"] = reports

    if "bias_reports" in st.session_state:
        for report in st.session_state["bias_reports"]:
            st.markdown(f"#### {report.topic}")
            status = "🔴 Potential bias detected" if report.is_flagged else "🟢 Consistent"
            st.write(f"{status} — {report.explanation}")
            case_df = pd.DataFrame([c.__dict__ for c in report.cases])
            st.dataframe(case_df[["variant", "query", "answer"]], use_container_width=True, hide_index=True)
            st.divider()
