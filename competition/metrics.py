"""
Metric computation functions for the GNN Molecular Graph Classification Challenge.
==================================================================================

All metric helpers live here so that evaluate.py, render_leaderboard.py,
and any notebook / script can ``from competition.metrics import …``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ------------------------------------------------------------------
# Primary metric
# ------------------------------------------------------------------

def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute macro-averaged F1 score (primary ranking metric)."""
    return float(f1_score(y_true, y_pred, average="macro"))


# ------------------------------------------------------------------
# Full metric suite
# ------------------------------------------------------------------

def compute_all_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, Any]:
    """Return a dictionary with every computed metric."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return {
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "accuracy": float(accuracy_score(yt, yp)),
        "precision_macro": float(
            precision_score(yt, yp, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(yt, yp, average="macro", zero_division=0)
        ),
        "f1_class_0": float(f1_score(yt, yp, pos_label=0, average="binary")),
        "f1_class_1": float(f1_score(yt, yp, pos_label=1, average="binary")),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
    }


# ------------------------------------------------------------------
# Efficiency metric
# ------------------------------------------------------------------

def efficiency_score(
    f1: float,
    inference_time_ms: float,
    total_params: int,
) -> float:
    """
    Efficiency = F1² / (log₁₀(time_ms) × log₁₀(params)).

    Higher is better — rewards both accuracy and computational economy.
    """
    if f1 <= 0 or inference_time_ms <= 0 or total_params <= 0:
        return 0.0
    t = max(inference_time_ms, 0.1)
    p = max(total_params, 100)
    log_t = math.log10(t)
    log_p = math.log10(p)
    denom = log_t * log_p
    if denom <= 0:
        denom = max(log_p, 1.0)
    return round((f1 ** 2) / denom, 6)
