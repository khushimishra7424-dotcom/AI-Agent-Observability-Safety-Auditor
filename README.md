# 🛡️ AI Agent Observability & Safety Auditor

An AI system that monitors, evaluates, and red-teams *another* AI system in real time.

This project audits a small RAG-based policy assistant (the **Target Agent**, built on
a company HR / product / loan-policy knowledge base) using four independent auditor
modules:

- 🔍 **Hallucination Detector** — flags answers not grounded in retrieved source documents,
  using real sentence embeddings (`all-MiniLM-L6-v2`) for semantic similarity, plus an
  **optional LLM-as-judge** second opinion when running in LLM mode
- ⚖️ **Bias / Fairness Checker** — paired counterfactual prompts to test consistency across demographics
- 🛡️ **Prompt Injection Red-Team** — fires adversarial prompts to test if the agent can be hijacked
- 📈 **Confidence Calibration** — tracks whether stated confidence matches real-world accuracy

### What changed in this version
- **Real embeddings instead of raw TF-IDF** for the hallucination/bias similarity checks
  (`core/embeddings.py`), with a safe automatic fallback to TF-IDF if the model can't be
  downloaded (e.g. no internet on first run).
- **LLM-as-judge**: when an `OPENAI_API_KEY` is set, a second lightweight LLM call
  independently scores how well each answer is grounded in the retrieved context, and
  is blended into the Hallucination Score.
- **Polished UI** (`core/theme.py`): gradient page headers, a live agent-mode status
  badge in the sidebar, styled metric cards, and quick stats on the home page.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # optional: add OPENAI_API_KEY to enable LLM mode
streamlit run app.py
```

If `OPENAI_API_KEY` is left empty, the target agent runs in **Local Mode**
(extractive answers, no external API calls, works fully offline).

## Pages

- **Live Audit** — ask the agent a question and see the full audit breakdown in real time
- **Dashboard** — historical trust score trends and flagged response analytics
- **Red Team Test** — run the full adversarial prompt injection + bias suite on demand
- **Calibration** — reliability curve and Expected Calibration Error (ECE)

## Project structure

```
ai-agent-auditor/
├── app.py
├── requirements.txt
├── .env.example
├── core/
│   ├── agent.py         # TargetAgent (RAG assistant being audited)
│   ├── vectorstore.py   # TF-IDF vector store
│   └── database.py      # SQLite persistence
├── auditor/
│   ├── hallucination.py
│   ├── bias.py
│   ├── injection.py
│   ├── calibration.py
│   └── trust_score.py
├── pages/
│   ├── 1_Live_Audit.py
│   ├── 2_Dashboard.py
│   ├── 3_Red_Team_Test.py
│   └── 4_Calibration.py
└── data/
    ├── knowledge_base/
    └── adversarial_prompts.json
```
