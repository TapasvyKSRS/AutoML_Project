"""
Stage 5: Multi-objective analysis (accuracy vs cost).

"""
from __future__ import annotations

import numpy as np

from automl.hpo import Trial


def pareto_front(history: list[Trial]) -> list[Trial]:
    """Return the non-dominated trials maximising R^2 while minimising time.

    A trial A dominates B if A.r2 >= B.r2 and A.train_time <= B.train_time
    with at least one strict inequality.
    """
    # only consider full-fidelity trials for a fair time comparison
    if not history:
        return []
    max_fid = max(t.fidelity for t in history)
    pts = [t for t in history if t.fidelity == max_fid]

    front: list[Trial] = []
    for a in pts:
        dominated = False
        for b in pts:
            if b is a:
                continue
            better_or_equal = (b.r2 >= a.r2) and (b.train_time <= a.train_time)
            strictly_better = (b.r2 > a.r2) or (b.train_time < a.train_time)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(a)

    # sort front by training time for nice plotting
    front.sort(key=lambda t: t.train_time)
    return front


def select_from_front(front: list[Trial], prefer: str = "accuracy") -> Trial:
    """Pick one config from the Pareto front.

    prefer = "accuracy" -> highest R^2 (default).
    prefer = "speed"    -> fastest config within an ABSOLUTE 0.01 R^2 of the best.
    This is the optional user-interaction window mentioned in the exam sheet.
    """
    if not front:
        raise ValueError("Empty Pareto front")

    if prefer == "speed":
        best_r2 = max(t.r2 for t in front)
        cheap = [t for t in front if t.r2 >= best_r2 - 0.01]
        return min(cheap, key=lambda t: t.train_time)

    return max(front, key=lambda t: t.r2)