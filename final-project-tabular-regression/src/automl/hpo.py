"""
Stage 4: Multi-fidelity Hyperparameter Optimization.

This module implements a self-contained multi-fidelity search that follows the
BOHB idea from the lecture:

  * Fidelity = fraction of the training data used to fit a model.
    Cheap evaluations (small fraction) let us screen many configs; the promising
    ones are re-evaluated on larger fractions (Successive Halving).
  * Configuration proposal starts as random search and switches to a simple
    model-based (Bayesian) proposer once enough observations exist, mirroring
    how BOHB replaces Hyperband's random sampling with a TPE-style model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from automl.models import build_model, build_pipeline
from automl.search_space import SEARCH_SPACES


@dataclass
class Trial:
    model_name: str
    config: dict[str, Any]
    fidelity: float          # fraction of training data used
    r2: float                # validation R^2
    train_time: float        # seconds to fit at this fidelity


@dataclass
class HPOResult:
    best_model_name: str
    best_config: dict[str, Any]
    best_r2: float
    history: list[Trial] = field(default_factory=list)


def _take(X, idx):
    """Row-select from a DataFrame or ndarray."""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _evaluate(model_name, config, X_tr, y_tr, X_val, y_val, fidelity, seed,
              cv_folds=1, n_jobs=1):
    """Score a config at a given fidelity (fraction of training data).

    IMPORTANT: X_tr is the RAW (untransformed) training data. We build a fresh
    sklearn Pipeline(preprocessor -> model) for every fit so the preprocessing
    is fitted only on that fold's training rows. Passing a pre-transformed
    matrix here would leak imputation/scaling/encoding statistics into the
    inner validation folds and inflate the scores.

    Fidelity subsets are NESTED: we draw one seeded permutation per call and
    take a prefix of it, so a larger fidelity is a strict superset of a smaller
    one. This makes successive-halving promotions comparable.

    Returns (mean_r2, total_train_time).
    """
    rng = np.random.RandomState(seed)
    n = X_tr.shape[0]
    perm = rng.permutation(n)                     # one permutation -> nested subsets
    size = min(n, max(50, int(fidelity * n)))
    idx = perm[:size]
    X_sub, y_sub = _take(X_tr, idx), y_tr[idx]

    # ---- single-split mode ----
    if cv_folds <= 1:
        pipe = build_pipeline(model_name, config, X_sub, seed=seed, n_jobs=n_jobs)
        t0 = time.time()
        pipe.fit(X_sub, y_sub)
        train_time = time.time() - t0
        return r2_score(y_val, pipe.predict(X_val)), train_time

    # ---- k-fold cross-validation mode (preprocessing inside each fold) ----
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_scores = []
    total_time = 0.0
    for train_i, val_i in kf.split(np.arange(len(y_sub))):
        X_f, y_f = _take(X_sub, train_i), y_sub[train_i]
        X_v, y_v = _take(X_sub, val_i), y_sub[val_i]
        pipe = build_pipeline(model_name, config, X_f, seed=seed, n_jobs=n_jobs)
        t0 = time.time()
        try:
            pipe.fit(X_f, y_f)
        except Exception as exc:                  # trial-level failure handling
            import logging
            logging.getLogger(__name__).warning(
                "Trial failed (%s, %s): %s", model_name, config, exc)
            return -np.inf, time.time() - t0
        total_time += time.time() - t0
        fold_scores.append(r2_score(y_v, pipe.predict(X_v)))

    return float(np.mean(fold_scores)), total_time


def _propose_config(space, history_for_model, rng, n_random=10):
    """Propose the next config.

    Cold start: random sampling. Warm phase: a lightweight Bayesian-style
    proposer that samples several random candidates and picks the one closest to
    the best-so-far config in normalised space (a cheap stand-in for a surrogate
    model's exploitation step). Replace with SMAC/NePS for a full BO loop.
    """
    if len(history_for_model) < n_random:
        return dict(space.sample_configuration())

    # exploit: bias sampling toward the incumbent
    best = max(history_for_model, key=lambda t: t.r2)
    candidates = [dict(space.sample_configuration()) for _ in range(16)]

    def distance(cfg):
        d = 0.0
        for k, v in cfg.items():
            bv = best.config[k]
            if isinstance(v, (int, float)) and isinstance(bv, (int, float)):
                scale = abs(bv) + 1e-9
                d += ((v - bv) / scale) ** 2
        return d

    # 70% of the time exploit (closest to incumbent), else explore (random)
    if rng.rand() < 0.7:
        return min(candidates, key=distance)
    return candidates[rng.randint(len(candidates))]


def optimize(
    model_names: list[str],
    X_tr, y_tr, X_val, y_val,
    n_iters: int = 60,
    fidelities: tuple[float, ...] = (0.1, 0.3, 1.0),
    eta: int = 3,
    seed: int = 0,
    cv_folds: int = 1,
) -> HPOResult:
    """Run multi-fidelity HPO over the given model families.

    Strategy (Successive-Halving flavoured):
      1. Sample a batch of configs, evaluate all at the lowest fidelity.
      2. Keep the top 1/eta, re-evaluate them at the next fidelity.
      3. Repeat until the highest fidelity; the best there is the incumbent.
      4. Loop until the iteration budget is exhausted.

    Parameters
    ----------
    model_names : which families to search (from the screening stage)
    n_iters     : total number of low-fidelity configs to sample across brackets
    fidelities  : increasing data fractions used as the fidelity ladder
    eta         : halving rate between fidelities
    """
    rng = np.random.RandomState(seed)
    spaces = {m: SEARCH_SPACES[m](seed=seed) for m in model_names}
    history: list[Trial] = []

    best = HPOResult(model_names[0], None, -np.inf, history)

    n_low = max(eta ** (len(fidelities) - 1), eta)
    brackets = max(1, n_iters // n_low)

    for _ in range(brackets):
        # ---- rung 0: many configs at the lowest fidelity ----
        survivors: list[tuple[str, dict]] = []
        for _ in range(n_low):
            m = model_names[rng.randint(len(model_names))]
            cfg = _propose_config(
                spaces[m], [t for t in history if t.model_name == m], rng
            )
            r2, tt = _evaluate(m, cfg, X_tr, y_tr, X_val, y_val, fidelities[0],
                               seed, cv_folds=cv_folds)
            history.append(Trial(m, cfg, fidelities[0], r2, tt))
            survivors.append((m, cfg, r2))

        # ---- promote through the fidelity ladder ----
        for rung, fidelity in enumerate(fidelities[1:], start=1):
            survivors.sort(key=lambda x: x[2], reverse=True)
            keep = max(1, len(survivors) // eta)
            survivors = survivors[:keep]

            promoted = []
            for m, cfg, _prev in survivors:
                r2, tt = _evaluate(m, cfg, X_tr, y_tr, X_val, y_val, fidelity,
                                   seed, cv_folds=cv_folds)
                history.append(Trial(m, cfg, fidelity, r2, tt))
                promoted.append((m, cfg, r2))

                if fidelity == fidelities[-1] and r2 > best.best_r2:
                    best.best_r2 = r2
                    best.best_model_name = m
                    best.best_config = cfg
            survivors = promoted

    # Fallback: if nothing reached the top fidelity (tiny budget), take best overall
    if best.best_config is None and history:
        top = max(history, key=lambda t: t.r2)
        best.best_model_name = top.model_name
        best.best_config = top.config
        best.best_r2 = top.r2

    return best