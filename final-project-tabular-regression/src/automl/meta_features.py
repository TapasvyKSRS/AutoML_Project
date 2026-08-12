"""
Stage 2: Meta-feature extraction (for algorithm selection / warm-starting).

Meta-features are cheap descriptors of a dataset. In the full "meta-learned"
version of the pipeline you can store these for the 5 practice datasets together
with the best config found on each, then at test time pick the nearest practice
dataset (in meta-feature space) to warm-start the search.

Even without warm-starting, logging these on the poster demonstrates the
algorithm-selection lecture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def extract_meta_features(X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    """Compute a small, robust set of dataset meta-features."""
    X_num = X.select_dtypes(include=["number"])

    n_rows, n_cols = X.shape
    n_numeric = X_num.shape[1]
    n_categorical = n_cols - n_numeric
    missing_ratio = float(X.isna().mean().mean())

    # Target statistics
    y = np.asarray(y, dtype=float)
    target_skew = float(pd.Series(y).skew()) if len(y) > 2 else 0.0
    target_std = float(np.std(y))

    # Mean absolute pairwise correlation among numeric features
    if n_numeric >= 2:
        corr = X_num.corr().abs().values
        # exclude the diagonal
        off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
        mean_abs_corr = float(np.nanmean(off_diag))
    else:
        mean_abs_corr = 0.0

    return {
        "n_rows": float(n_rows),
        "n_cols": float(n_cols),
        "n_numeric": float(n_numeric),
        "n_categorical": float(n_categorical),
        "missing_ratio": missing_ratio,
        "target_skew": target_skew,
        "target_std": target_std,
        "mean_abs_feature_corr": mean_abs_corr,
        "rows_to_cols_ratio": float(n_rows) / float(max(n_cols, 1)),
    }