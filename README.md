🛡️ AI Agent Observability & Safety Auditor
An advanced Streamlit dashboard that monitors, evaluates, and red-teams AI systems in real time.
Instead of building just an AI agent, this project builds the safety & observability layer that sits on top of it — answering the critical question:
👉 "How do we know when our AI is wrong, biased, or being manipulated?"

🔑 Key Features
Hallucination Detector 🔍 — flags ungrounded answers using embeddings + optional LLM judge

Bias / Fairness Checker ⚖️ — tests consistency across demographics with paired counterfactual prompts

Prompt Injection Red-Team 🛡️ — adversarial suite to check resilience against hijacking

Confidence Calibration 📈 — tracks whether confidence scores match real-world accuracy

📊 Dashboard Modules
Live Audit — ask questions and see full audit breakdown in real time

Historical Dashboard — trust score trends & flagged response analytics

Red Team Test — run adversarial prompt injection suite on demand

Calibration — reliability curve & Expected Calibration Error (ECE)

📚 Knowledge Base
Currently loaded for the Target Agent:

HR leave policy (hr_policy.txt)

CloudSync product FAQ (product_faq.txt)

Loan eligibility policy (loan_policy.txt)

🚀 Tech Stack
Python + Streamlit

RAG pipeline with embeddings

Custom auditor modules for safety & trustworthiness
