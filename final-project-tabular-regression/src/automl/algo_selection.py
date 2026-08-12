"""
Stage 3: Algorithm screening.

We quick-fit each candidate model (with default hyperparameters) on a *subset*
of the training data and rank them by validation R^2. The top-k families are
then passed to the expensive multi-fidelity HPO stage.
"""
from __future__ import annotations

import logging
import numpy as np
from sklearn.metrics import r2_score

from automl.models import build_model, build_pipeline
from automl.search_space import SEARCH_SPACES

logger = logging.getLogger(__name__)


def screen_algorithms(
    X_train, y_train, X_val, y_val,
    subset_fraction: float = 0.3,
    top_k: int = 2,
    seed: int = 0,
) -> list[str]:
    """Fit each model family on a data subset and return the top-k by val R^2.

    Parameters
    ----------
    X_train, y_train : RAW training features / targets (preprocessing is
                       fitted inside, so no leakage)
    X_val, y_val     : RAW validation features / targets
    subset_fraction  : fraction of training rows to use for the quick fit
    top_k            : how many model families to keep for full HPO
    seed             : reproducibility
    """
    rng = np.random.RandomState(seed)
    n = X_train.shape[0]
    subset_size = max(50, int(subset_fraction * n))
    idx = rng.choice(n, size=min(subset_size, n), replace=False)
    X_sub = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
    y_sub = y_train[idx]

    scores: dict[str, float] = {}
    for model_name in SEARCH_SPACES:
        space = SEARCH_SPACES[model_name](seed=seed)
        default_cfg = dict(space.get_default_configuration())
        pipe = build_pipeline(model_name, default_cfg, X_sub, seed=seed)
        try:
            pipe.fit(X_sub, y_sub)
            scores[model_name] = r2_score(y_val, pipe.predict(X_val))
        except Exception as exc:
            logger.warning("Screening failed for %s: %s", model_name, exc)
            scores[model_name] = float("-inf")

    ranked = sorted(scores, key=lambda m: scores[m], reverse=True)
    return ranked[:top_k], scores