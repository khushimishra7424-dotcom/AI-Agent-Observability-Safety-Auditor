"""
Combines individual audit signals into one 0-100 Trust Score per response.
"""

from dataclasses import dataclass


@dataclass
class TrustScoreBreakdown:
    trust_score: float
    groundedness_component: float
    confidence_component: float
    label: str


def compute_trust_score(confidence: float, hallucination_score: float) -> TrustScoreBreakdown:
    groundedness = 1.0 - hallucination_score
    raw_score = 0.65 * groundedness + 0.35 * confidence
    trust_score = round(raw_score * 100, 1)

    if trust_score >= 75:
        label = "High Trust"
    elif trust_score >= 45:
        label = "Medium Trust"
    else:
        label = "Low Trust"

    return TrustScoreBreakdown(
        trust_score=trust_score,
        groundedness_component=round(groundedness * 100, 1),
        confidence_component=round(confidence * 100, 1),
        label=label,
    )
