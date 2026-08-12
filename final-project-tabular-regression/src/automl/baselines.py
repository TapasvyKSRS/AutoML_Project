"""
Baseline optimizers

Two baselines:
  * default_baseline
  * random_search

Both return the same HPOResult
"""
from __future__ import annotations

import time
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from automl.models import build_model, build_pipeline
from automl.search_space import SEARCH_SPACES
from automl.hpo import Trial, HPOResult


def _take(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _cv_score(model_name, config, X_tr, y_tr, cv_folds, seed, n_jobs=1):
    """k-fold CV on RAW data; preprocessing is fitted inside each fold."""
    kf = KFold(n_splits=max(2, cv_folds), shuffle=True, random_state=seed)
    scores, total_time = [], 0.0
    for tr_i, val_i in kf.split(np.arange(len(y_tr))):
        X_f, y_f = _take(X_tr, tr_i), y_tr[tr_i]
        X_v, y_v = _take(X_tr, val_i), y_tr[val_i]
        pipe = build_pipeline(model_name, config, X_f, seed=seed, n_jobs=n_jobs)
        t0 = time.time()
        try:
            pipe.fit(X_f, y_f)
        except Exception:
            return -np.inf, time.time() - t0
        total_time += time.time() - t0
        scores.append(r2_score(y_v, pipe.predict(X_v)))
    return float(np.mean(scores)), total_time


def default_baseline(
    model_names, X_tr, y_tr, X_val, y_val, cv_folds=3, seed=0
) -> HPOResult:
    """Fit each family with default hyperparameters; keep the best."""
    history: list[Trial] = []
    best = HPOResult(model_names[0], None, -np.inf, history)
    for model_name in model_names:
        space = SEARCH_SPACES[model_name](seed=seed)
        cfg = dict(space.get_default_configuration())
        score, ttime = _cv_score(model_name, cfg, X_tr, y_tr, cv_folds, seed)
        history.append(Trial(model_name, cfg, 1.0, score, ttime))
        if score > best.best_r2:
            best.best_r2 = score
            best.best_model_name = model_name
            best.best_config = cfg
    return best


def random_search(
    model_names, X_tr, y_tr, X_val, y_val, n_iters=60, cv_folds=3, seed=0
) -> HPOResult:
    """Uniform random sampling over the search spaces; keep the best."""
    rng = np.random.RandomState(seed)
    spaces = {m: SEARCH_SPACES[m](seed=seed) for m in model_names}
    history: list[Trial] = []
    best = HPOResult(model_names[0], None, -np.inf, history)

    for _ in range(n_iters):
        m = model_names[rng.randint(len(model_names))]
        cfg = dict(spaces[m].sample_configuration())
        score, ttime = _cv_score(m, cfg, X_tr, y_tr, cv_folds, seed)
        history.append(Trial(m, cfg, 1.0, score, ttime))
        if score > best.best_r2:
            best.best_r2 = score
            best.best_model_name = m
            best.best_config = cfg
    return best