"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns dummy predictions.
You do not need to use this setup or sklearn and you can modify this however you like.
"""
from __future__ import annotations
 
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import logging
 
# --- pipeline stage imports ---
from automl.preprocessing import build_preprocessor, as_dataframe
from automl.meta_features import extract_meta_features
from automl.algo_selection import screen_algorithms
from automl.hpo import optimize
from automl.multi_objective import pareto_front, select_from_front
from automl.models import build_model, build_pipeline
 
logger = logging.getLogger(__name__)
 
METRICS = {"r2": r2_score}
 
 
class AutoML:
 
    def __init__(
        self,
        seed: int,
        metric: str = "r2",
        n_iters: int = 60,
        prefer: str = "accuracy",
        cv_folds: int = 3,
        top_k: int = 3,
        backend: str = "builtin",
        portfolio_path: str | None = None,
    ) -> None:
        self.seed = seed
        self.metric = METRICS[metric]
        self._model: DummyRegressor | None = None
 
        # --- pipeline settings / state  ---
        self.n_iters = n_iters          # HPO budget (low-fidelity configs sampled)
        self.prefer = prefer            # Pareto pick: "accuracy" or "speed"
        self.cv_folds = cv_folds        # k-fold CV inside HPO (1 = single split)
        self.top_k = top_k              # families kept by screening for HPO
        if backend not in ("builtin", "smac", "bohb"):
            raise ValueError(
                f"backend must be 'builtin', 'smac' or 'bohb', got {backend!r}")
        if prefer not in ("accuracy", "speed"):
            raise ValueError(f"prefer must be 'accuracy' or 'speed', got {prefer!r}")
        self.backend = backend          # "builtin" (hpo.py) or "smac" (hpo_smac.py)
        self.portfolio_path = portfolio_path  # meta-learning warm-start portfolio
        self._preprocessor = None       # fitted preprocessing transformer
        self._best_model_name = None    # chosen model family
        self._best_config = None        # chosen hyperparameters
        self.summary_: dict = {}        # run summary for the poster
 
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> AutoML:
        X = as_dataframe(X)
        y = np.asarray(y).ravel()
 
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            random_state=self.seed,
            test_size=0.2,
        )
 
        # Stage 2: meta-features (logged for the poster / warm-starting)
        meta = extract_meta_features(X, y)
        logger.info(f"Meta-features: {meta}")
 
        # Stage 1: preprocessing (fit on the train split only)
        prep = build_preprocessor(X)
        X_train_p = prep.fit_transform(X_train)
        X_val_p = prep.transform(X_val)
 
        # Stage 3: algorithm screening (keep the top-2 model families)
        top_models, screen_scores = screen_algorithms(
            X_train_p, y_train, X_val_p, y_val,
            subset_fraction=0.3, top_k=self.top_k, seed=self.seed,
        )
        logger.info(f"Screening scores: {screen_scores}")
        logger.info(f"Top models for HPO: {top_models}")
 
        # Stage 4: multi-fidelity HPO (fidelity = fraction of training data)
        # Backend can be the built-in multi-fidelity search (hpo.py) or real
        # SMAC Bayesian optimisation (hpo_smac.py)
        if self.backend == "bohb":
            from automl.hpo_bohb import optimize_bohb
            result = optimize_bohb(
                model_names=top_models,
                X_tr=X_train, y_tr=y_train, X_val=X_val, y_val=y_val,
                n_iters=self.n_iters,
                cv_folds=self.cv_folds,
                seed=self.seed,
            )
        elif self.backend == "smac":
            from automl.hpo_smac import optimize_smac
 
            warm = []
            if self.portfolio_path:
                from automl.meta_warmstart import warmstart_configs, nearest_datasets
                warm = warmstart_configs(self.portfolio_path, meta, k=2)
                if warm:
                    near = nearest_datasets(self.portfolio_path, meta, k=2)
                    logger.info(f"Warm-starting from nearest datasets: {near}")
 
            result = optimize_smac(
                model_names=top_models,
                X_tr=X_train, y_tr=y_train, X_val=X_val, y_val=y_val,
                n_iters=self.n_iters,
                cv_folds=self.cv_folds,
                seed=self.seed,
                warmstart_configs=warm,
            )
        else:
            result = optimize(
                model_names=top_models,
                X_tr=X_train, y_tr=y_train, X_val=X_val, y_val=y_val,
                n_iters=self.n_iters,
                fidelities=(0.5, 1.0),
                eta=3,
                seed=self.seed,
                cv_folds=self.cv_folds,
            )
 
        # Stage 5: multi-objective selection (Pareto front, accuracy vs cost)
        # The Pareto front is kept for analysis/plots and for the optional
        # "speed" preference. For the default accuracy objective we take the
        # config that scored best at FULL fidelity, which is the reliable choice.
        front = pareto_front(result.history)
        if self.prefer == "speed" and front:
            chosen = select_from_front(front, prefer="speed")
            self._best_model_name = chosen.model_name
            self._best_config = chosen.config
        else:
            self._best_model_name = result.best_model_name
            self._best_config = result.best_config

        val_pipe = build_pipeline(self._best_model_name, self._best_config,
                                  X_train, seed=self.seed)
        val_pipe.fit(X_train, y_train)
        val_score = self.metric(y_val, val_pipe.predict(X_val))
        logger.info(f"Validation score (honest, holdout): {val_score:.4f}")

        # Inner-CV score from HPO, reported alongside for transparency.
        cv_score = result.best_r2
        logger.info(f"Inner-CV score from HPO: {cv_score:.4f}")

        # Stage 6: refit on ALL training data for the final model. No validation
        # score is computed from this model.
        self._model = build_pipeline(self._best_model_name, self._best_config,
                                     X, seed=self.seed)
        self._model.fit(X, y)

        # Save a summary
        self.summary_ = {
            "best_model": self._best_model_name,
            "best_config": self._best_config,
            "val_r2_holdout": val_score,
            "cv_r2_inner": cv_score,
            "screening_scores": screen_scores,
            "meta_features": meta,
            "pareto_front_size": len(front),
            "n_iters": self.n_iters,
            "seed": self.seed,
        }
 
        return self
 
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted")
 
        return self._model.predict(as_dataframe(X))