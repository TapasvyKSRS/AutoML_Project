"""
Model factory: turns a (model_name, config dict) pair into an untrained,
sklearn-compatible estimator, or a Pipeline that includes preprocessing.

Keeping this in one place means the HPO loop, the algorithm-screening stage and
the final-training stage all build models identically.

Optional dependencies (catboost, tabpfn) are imported defensively: if a library
is missing the corresponding family is simply absent from AVAILABLE_MODELS and
the rest of the pipeline runs unchanged.
"""
from __future__ import annotations

import ssl
from typing import Any

_orig_load_default_certs = ssl.SSLContext.load_default_certs


def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    """Load trust roots, tolerating a malformed Windows certificate store.

    If the Windows store raises, we fall back to certifi's CA bundle so the
    context still has valid trust roots. Simply swallowing the error would
    leave the context with NO certificates, which makes every HTTPS request
    fail later with a confusing downstream error (TabPFN, for instance,
    misreports it as a 'gated repository' problem).
    """
    try:
        return _orig_load_default_certs(self, purpose)
    except (ssl.SSLError, OSError):
        pass

    # Fall back to a known-good CA bundle.
    try:
        import certifi
        self.load_verify_locations(cafile=certifi.where())
    except Exception:
        pass

    # SSL_CERT_FILE / SSL_CERT_DIR if the user set them.
    try:
        self.set_default_verify_paths()
    except Exception:
        pass
    return None


ssl.SSLContext.load_default_certs = _safe_load_default_certs
# ---------------------------------------------------------------------------

from lightgbm import LGBMRegressor                                    
from sklearn.ensemble import (                                        
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge     

# Families that are always available.
AVAILABLE_MODELS = {
    "ridge",
    "elasticnet",
    "bayesian_ridge",
    "random_forest",
    "extra_trees",
    "lightgbm",
    "hist_gradient_boosting",
}

# --- optional: CatBoost -----------------------------------------------------
try:
    from catboost import CatBoostRegressor
    AVAILABLE_MODELS.add("catboost")
except Exception:                                  
    CatBoostRegressor = None

# --- optional: TabPFN -------------------------------------------------------
try:
    from tabpfn import TabPFNRegressor
    AVAILABLE_MODELS.add("tabpfn")
except Exception:                                    # pragma: no cover
    TabPFNRegressor = None


def build_pipeline(model_name: str, config: dict[str, Any], X, seed: int = 0,
                   n_jobs: int = 1):
    """Return an sklearn Pipeline of (preprocessing -> model).

    Using a Pipeline means the preprocessor is fitted INSIDE each CV fold, so
    imputation/scaling/encoding statistics never see the inner validation fold.
    """
    from sklearn.pipeline import Pipeline
    from automl.preprocessing import build_preprocessor, as_dataframe

    X = as_dataframe(X)
    return Pipeline([
        ("prep", build_preprocessor(X)),
        ("model", build_model(model_name, config, seed=seed, n_jobs=n_jobs)),
    ])


def build_model(model_name: str, config: dict[str, Any], seed: int = 0,
                n_jobs: int = 1):
    """Instantiate an unfitted estimator from a model name and config dict."""
    cfg = dict(config)          # copy so we never mutate the caller's dict

    # ---------------- linear / regularised ----------------
    if model_name == "ridge":
        return Ridge(random_state=seed, **cfg)

    if model_name == "elasticnet":
        return ElasticNet(random_state=seed, max_iter=5000, **cfg)

    if model_name == "bayesian_ridge":
        # BayesianRidge has no random_state (it is deterministic).
        return BayesianRidge(**cfg)

    # ---------------- bagging ensembles ----------------
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **cfg)

    if model_name == "extra_trees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs, **cfg)

    # ---------------- boosting ensembles ----------------
    if model_name == "lightgbm":
        # LightGBM ignores `subsample` unless bagging is actually scheduled.
        # Without subsample_freq >= 1 the HPO wastes trials on a dead knob.
        if cfg.get("subsample", 1.0) < 1.0:
            cfg["subsample_freq"] = 1
        return LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbose=-1, **cfg)

    if model_name == "hist_gradient_boosting":
        # No n_jobs parameter; parallelism is controlled by OMP_NUM_THREADS.
        return HistGradientBoostingRegressor(random_state=seed, **cfg)

    if model_name == "catboost":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed")
        return CatBoostRegressor(
            random_seed=seed, thread_count=n_jobs, verbose=0,
            allow_writing_files=False, **cfg,
        )

    # ---------------- pretrained / in-context ----------------
    if model_name == "tabpfn":
        if TabPFNRegressor is None:
            raise ImportError("tabpfn is not installed")
        return TabPFNRegressor(device="cpu", **cfg)

    raise ValueError(f"Unknown model_name: {model_name}")