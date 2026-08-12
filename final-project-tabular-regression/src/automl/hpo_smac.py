"""
Bayesian optimisation backend using SMAC3.

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

FAILED_TRIAL_SCORE = -1e6
_FAIL_JITTER = 1e-3


def _failed_score(n_seen: int) -> float:
    return FAILED_TRIAL_SCORE - (n_seen % 97) * _FAIL_JITTER


def _take(X, idx):
    """Row-select from a DataFrame or ndarray."""
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


def _score_config(model_name, config, X_tr, y_tr, cv_folds, seed, n_jobs=1):
    """k-fold CV score on RAW data; preprocessing is fitted inside each fold."""
    kf = KFold(n_splits=max(2, cv_folds), shuffle=True, random_state=seed)
    scores, total_time = [], 0.0
    for tr_i, val_i in kf.split(np.arange(len(y_tr))):
        X_f, y_f = _take(X_tr, tr_i), y_tr[tr_i]
        X_v, y_v = _take(X_tr, val_i), y_tr[val_i]
        pipe = build_pipeline(model_name, config, X_f, seed=seed, n_jobs=n_jobs)
        t0 = time.time()
        try:
            pipe.fit(X_f, y_f)
        except Exception as exc:
            logger.warning("Trial failed (%s): %s", model_name, exc)
            return None, time.time() - t0
        total_time += time.time() - t0
        scores.append(r2_score(y_v, pipe.predict(X_v)))
    mean_score = float(np.mean(scores))
    if not np.isfinite(mean_score):
        logger.warning("Non-finite score for %s; treating as failed trial", model_name)
        return None, total_time
    return mean_score, total_time


def optimize_smac(
    model_names: list[str],
    X_tr, y_tr, X_val, y_val,
    n_iters: int = 60,
    cv_folds: int = 3,
    seed: int = 0,
    warmstart_configs: list[dict] | None = None,
    n_jobs: int = 1,
) -> HPOResult:
    """Run SMAC Bayesian optimisation once per model family, keep the global best.

    Each family has its own ConfigurationSpace, so they get separate studies and
    the budget is split evenly between them.

    Parameters
    ----------
    n_iters : total trial budget across all families.
    cv_folds : inner CV folds used to score each configuration.
    warmstart_configs : optional configs (e.g. from the meta-learning portfolio)
        injected into SMAC's initial design. Configs that do not fit a given
        family's space are skipped.
    n_jobs : threads per model fit. Keep at 1 during search to avoid
        oversubscription across trials.
    """
    from smac import HyperparameterOptimizationFacade, Scenario
    from ConfigSpace import Configuration

    history: list[Trial] = []
    best = HPOResult(model_names[0], None, -np.inf, history)

    per_model_budget = max(1, n_iters // len(model_names))

    for model_name in model_names:
        space = SEARCH_SPACES[model_name](seed=seed)
        def target(config: Configuration, seed: int = seed) -> float:
            cfg = dict(config)
            score, ttime = _score_config(
                model_name, cfg, X_tr, y_tr, cv_folds, seed, n_jobs
            )
            if score is None or not np.isfinite(score):
                score = _failed_score(len(history))
            history.append(Trial(model_name, cfg, 1.0, score, ttime))
            return -score          

        scenario = Scenario(
            space,
            deterministic=True,
            n_trials=per_model_budget,
            seed=seed,
        )

        # Inject warm-start configurations into the initial design if given.
        additional = []
        for c in (warmstart_configs or []):
            try:
                additional.append(Configuration(space, values=c))
            except Exception:
                pass          

        smac_kwargs = dict(scenario=scenario, target_function=target,
                           overwrite=True)
        if additional:
            try:
                from smac.initial_design import DefaultInitialDesign
                smac_kwargs["initial_design"] = DefaultInitialDesign(
                    scenario, additional_configs=additional
                )
                logger.info("Warm-starting %s with %d config(s)",
                            model_name, len(additional))
            except Exception as exc:
                logger.warning(
                    "Warm-start unavailable for %s (%s); using SMAC's default "
                    "initial design instead.", model_name, exc)

        try:
            smac = HyperparameterOptimizationFacade(**smac_kwargs)
            incumbent = smac.optimize()
        except Exception as exc:
            logger.warning("SMAC aborted for %s (%s); skipping this family.",
                           model_name, exc)
            continue

        inc_cfg = dict(incumbent)
        try:
            inc_score = -float(smac.runhistory.get_cost(incumbent))
        except Exception:
            inc_score, _ = _score_config(
                model_name, inc_cfg, X_tr, y_tr, cv_folds, seed, n_jobs)
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