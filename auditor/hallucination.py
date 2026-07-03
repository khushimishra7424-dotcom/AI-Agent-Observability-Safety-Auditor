"""
HallucinationDetector - flags answers not grounded in retrieved context.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from core.vectorstore import VectorStore, RetrievalResult

HEDGE_PATTERNS = [
    r"best guess",
    r"cannot confirm",
    r"general knowledge",
    r"not certain",
    r"i think",
    r"probably",
    r"as far as i know",
    r"typically depends",
]


@dataclass
class HallucinationReport:
    hallucination_score: float
    is_flagged: bool
    semantic_overlap: float
    retrieval_strength: float
    hedge_detected: bool
    explanation: str
    llm_judge_score: Optional[float] = None


class HallucinationDetector:
    """
    Scores how grounded an answer is in the retrieved context.

    Groundedness signal is a blend of:
      - semantic_overlap: embedding similarity between answer and context
      - retrieval_strength: how relevant the top retrieved chunk was
      - (optional) llm_judge_score: a second LLM call that independently
        rates groundedness, used only when an OpenAI client is supplied
        (i.e. the target agent is running in LLM mode)
    """

    def __init__(self, store: VectorStore, threshold: float = 0.45, llm_client=None,
                 llm_model: str = "gpt-4o-mini"):
        self.store = store
        self.threshold = threshold
        self.llm_client = llm_client
        self.llm_model = llm_model

    def check(self, answer: str, retrieved: List[RetrievalResult]) -> HallucinationReport:
        context = " ".join(r.chunk.text for r in retrieved)
        semantic_overlap = self.store.similarity(answer, context) if context.strip() else 0.0
        retrieval_strength = max((r.score for r in retrieved), default=0.0)
        hedge_detected = any(re.search(p, answer, re.IGNORECASE) for p in HEDGE_PATTERNS)

        llm_judge_score = None
        if self.llm_client is not None and context.strip() and answer.strip():
            llm_judge_score = self._llm_judge(answer, context)

        if llm_judge_score is not None:
            groundedness = 0.4 * semantic_overlap + 0.2 * retrieval_strength + 0.4 * llm_judge_score
        else:
            groundedness = 0.6 * semantic_overlap + 0.4 * retrieval_strength

        hallucination_score = 1.0 - groundedness
        if hedge_detected:
            hallucination_score = min(1.0, hallucination_score + 0.2)

        is_flagged = hallucination_score >= self.threshold

        if is_flagged:
            reasons = []
            if semantic_overlap < 0.3:
                reasons.append("low semantic overlap between answer and retrieved context")
            if retrieval_strength < 0.3:
                reasons.append("weak retrieval match for the query (no strongly relevant document)")
            if hedge_detected:
                reasons.append("hedging/guessing language detected in the answer")
            if llm_judge_score is not None and llm_judge_score < 0.4:
                reasons.append("LLM judge independently rated the answer as poorly grounded")
            explanation = "Flagged: " + "; ".join(reasons) if reasons else "Flagged: low overall groundedness"
        else:
            explanation = "Answer appears well-grounded in retrieved context."

        return HallucinationReport(
            hallucination_score=round(hallucination_score, 3),
            is_flagged=is_flagged,
            semantic_overlap=round(semantic_overlap, 3),
            retrieval_strength=round(retrieval_strength, 3),
            hedge_detected=hedge_detected,
            explanation=explanation,
            llm_judge_score=round(llm_judge_score, 3) if llm_judge_score is not None else None,
        )

    def _llm_judge(self, answer: str, context: str) -> Optional[float]:
        """Ask a second LLM call to independently rate groundedness 0-1."""
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict fact-checking judge. Given CONTEXT and ANSWER, "
                            "output only a single number between 0 and 1 representing how "
                            "fully the ANSWER is supported by CONTEXT (1 = fully grounded, "
                            "0 = not grounded / hallucinated). Output only the number."
                        ),
                    },
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nScore:"},
                ],
                temperature=0,
                max_tokens=5,
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r"(\d*\.?\d+)", text)
            if not match:
                return None
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        except Exception:
            return None
