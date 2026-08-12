"""
BOHB backend: Bayesian optimisation + Hyperband, via SMAC's MultiFidelityFacade.

                     random proposals     model-based proposals
    full budget      random_search        SMAC        (hpo_smac.py)
    Hyperband        Hyperband            BOHB        (this file

Fidelity dimension: fraction of training rows (expressed as a percentage budget,
because SMAC expects numeric budgets). budget=100 means the full training split.

"""
from __future__ import annotations

import logging
import time

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from automl.models import build_pipeline
from automl.search_space import SEARCH_SPACES
from automl.hpo import Trial, HPOResult

logger = logging.getLogger(__name__)


# Finite penalty for failed trials -- an infinite cost makes SMAC's surrogate
# produce NaN and crash ("Input y contains NaN").
FAILED_TRIAL_SCORE = -1e6
_FAIL_JITTER = 1e-3


def _failed_score(n_seen: int) -> float:
    # Identical failure costs collapse SMAC's min-max normalisation to NaN.
    return FAILED_TRIAL_SCORE - (n_seen % 97) * _FAIL_JITTER


def _take(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _score_at_budget(model_name, config, X_tr, y_tr, budget, cv_folds, seed,
                     n_jobs=1):
    """k-fold CV score using `budget` percent of the training rows.

    Subsets are nested: one seeded permutation, prefix taken by budget, so a
    promoted configuration is re-evaluated on a strict superset of its earlier
    sample. Preprocessing is fitted inside every fold (no leakage).
    """
    rng = np.random.RandomState(seed)
    n = X_tr.shape[0]
    perm = rng.permutation(n)
    size = min(n, max(50, int(n * float(budget) / 100.0)))
    idx = perm[:size]
    X_sub, y_sub = _take(X_tr, idx), y_tr[idx]

    kf = KFold(n_splits=max(2, cv_folds), shuffle=True, random_state=seed)
    scores, total_time = [], 0.0
    for tr_i, val_i in kf.split(np.arange(len(y_sub))):
        X_f, y_f = _take(X_sub, tr_i), y_sub[tr_i]
        X_v, y_v = _take(X_sub, val_i), y_sub[val_i]
        pipe = build_pipeline(model_name, config, X_f, seed=seed, n_jobs=n_jobs)
        t0 = time.time()
        try:
            pipe.fit(X_f, y_f)
        except Exception:
            return None, time.time() - t0
        total_time += time.time() - t0
        scores.append(r2_score(y_v, pipe.predict(X_v)))
    mean_score = float(np.mean(scores))
    if not np.isfinite(mean_score):
        return None, total_time
    return mean_score, total_time


def optimize_bohb(
    model_names: list[str],
    X_tr, y_tr, X_val, y_val,
    n_iters: int = 60,
    cv_folds: int = 3,
    seed: int = 0,
    min_budget: float = 25.0,
    max_budget: float = 100.0,
    eta: int = 3,
    n_jobs: int = 1,
) -> HPOResult:
    """Run BOHB (SMAC MultiFidelityFacade) for each model family.

    Parameters
    ----------
    min_budget, max_budget : percentage of training rows used at the lowest and
        highest fidelity rung.
    eta : Hyperband halving rate.
    """
    from smac import MultiFidelityFacade, Scenario
    from smac.intensifier.hyperband import Hyperband

    history: list[Trial] = []
    best = HPOResult(model_names[0], None, -np.inf, history)
    per_model_budget = max(1, n_iters // len(model_names))

    for model_name in model_names:
        space = SEARCH_SPACES[model_name](seed=seed)

        def target(config, seed: int = seed, budget: float = max_budget) -> float:
            cfg = dict(config)
            score, ttime = _score_at_budget(
                model_name, cfg, X_tr, y_tr, budget, cv_folds, seed, n_jobs)
            if score is None or not np.isfinite(score):
                score = _failed_score(len(history))
            history.append(Trial(model_name, cfg, float(budget) / 100.0,
                                 score, ttime))
            return -score            # SMAC minimises

        scenario = Scenario(
            space,
            deterministic=True,
            n_trials=per_model_budget,
            min_budget=min_budget,
            max_budget=max_budget,
            seed=seed,
        )
        try:
            intensifier = Hyperband(scenario, eta=eta)
            smac = MultiFidelityFacade(
                scenario, target, intensifier=intensifier, overwrite=True,
                logging_level=logging.WARNING)
            incumbent = smac.optimize()
        except BaseException as exc:
            import traceback
            msg = f"[BOHB] '{model_name}' aborted: {type(exc).__name__}: {exc}"
            print(msg, flush=True)
            traceback.print_exc()
            logger.error(msg)
            continue

        inc_cfg = dict(incumbent)
        inc_score, _ = _score_at_budget(
            model_name, inc_cfg, X_tr, y_tr, max_budget, cv_folds, seed, n_jobs)
        if inc_score is None:
            inc_score = FAILED_TRIAL_SCORE
        if inc_score > best.best_r2:
            best.best_r2 = inc_score
            best.best_model_name = model_name
            best.best_config = inc_cfg

    if best.best_config is None and history:
        top = max(history, key=lambda t: t.r2)
        best.best_model_name = top.model_name
        best.best_config = top.config
        best.best_r2 = top.r2

    return best