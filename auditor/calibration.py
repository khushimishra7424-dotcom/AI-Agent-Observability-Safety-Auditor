"""
CalibrationTracker - reliability curve + Expected Calibration Error.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np


@dataclass
class CalibrationResult:
    bin_edges: List[float]
    bin_confidence: List[float]
    bin_accuracy: List[float]
    bin_counts: List[int]
    ece: float
    n_samples: int


def compute_calibration(pairs: List[Dict[str, Any]], n_bins: int = 5) -> CalibrationResult:
    if not pairs:
        edges = list(np.linspace(0, 1, n_bins + 1))
        return CalibrationResult(edges, [0.0] * n_bins, [0.0] * n_bins, [0] * n_bins, 0.0, 0)

    confidences = np.array([p["confidence"] for p in pairs])
    correctness = np.array([1.0 if p["user_feedback"] == "correct" else 0.0 for p in pairs])

    edges = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc, bin_counts = [], [], []
    ece = 0.0
    n = len(pairs)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count > 0:
            avg_conf = float(confidences[mask].mean())
            avg_acc = float(correctness[mask].mean())
            ece += (count / n) * abs(avg_conf - avg_acc)
        else:
            avg_conf, avg_acc = 0.0, 0.0
        bin_conf.append(round(avg_conf, 3))
        bin_acc.append(round(avg_acc, 3))

    return CalibrationResult(
        bin_edges=list(edges),
        bin_confidence=bin_conf,
        bin_accuracy=bin_acc,
        bin_counts=bin_counts,
        ece=round(float(ece), 3),
        n_samples=n,
    )
