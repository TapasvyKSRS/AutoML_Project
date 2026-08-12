"""
Meta-learning warm-start (the novelty component).

"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The meta-feature keys we align on (must match meta_features.extract_meta_features)
META_KEYS = [
    "n_rows", "n_cols", "n_numeric", "n_categorical",
    "missing_ratio", "target_skew", "target_std",
    "mean_abs_feature_corr", "rows_to_cols_ratio",
]


def _vec(meta: dict) -> np.ndarray:
    return np.array([meta[k] for k in META_KEYS], dtype=float)


def save_to_portfolio(
    portfolio_path: Path,
    dataset_name: str,
    meta: dict,
    best_model: str,
    best_config: dict,
    score: float,
) -> None:
    """Append one dataset's result to the meta-knowledge portfolio (JSON)."""
    portfolio_path = Path(portfolio_path)
    if portfolio_path.exists():
        portfolio = json.loads(portfolio_path.read_text())
    else:
        portfolio = []

    # replace an existing entry for the same dataset if present
    portfolio = [e for e in portfolio if e["dataset"] != dataset_name]
    portfolio.append({
        "dataset": dataset_name,
        "meta": {k: meta[k] for k in META_KEYS},
        "best_model": best_model,
        "best_config": best_config,
        "score": score,
    })
    portfolio_path.write_text(json.dumps(portfolio, indent=2, default=str))


def warmstart_configs(
    portfolio_path: Path,
    meta: dict,
    k: int = 2,
) -> list[dict]:
    """Return best configs from the k nearest datasets in meta-feature space.

    Distances are computed on standardised meta-features so that large-magnitude
    features (like n_rows) don't dominate. Returns an empty list if the portfolio
    is missing or empty (so the caller falls back to normal search).
    """
    portfolio_path = Path(portfolio_path)
    if not portfolio_path.exists():
        return []

    portfolio = json.loads(portfolio_path.read_text())
    if not portfolio:
        return []

    # Build the matrix of stored meta-vectors and standardise.
    stored = np.array([_vec(e["meta"]) for e in portfolio])
    mu = stored.mean(axis=0)
    sigma = stored.std(axis=0) + 1e-9
    stored_z = (stored - mu) / sigma
    query_z = (_vec(meta) - mu) / sigma

    dists = np.linalg.norm(stored_z - query_z, axis=1)
    order = np.argsort(dists)[:k]

    return [portfolio[i]["best_config"] for i in order]


def nearest_datasets(
    portfolio_path: Path,
    meta: dict,
    k: int = 2,
) -> list[str]:
    """Names of the k nearest datasets (for logging / poster analysis)."""
    portfolio_path = Path(portfolio_path)
    if not portfolio_path.exists():
        return []
    portfolio = json.loads(portfolio_path.read_text())
    if not portfolio:
        return []
    stored = np.array([_vec(e["meta"]) for e in portfolio])
    mu, sigma = stored.mean(axis=0), stored.std(axis=0) + 1e-9
    stored_z = (stored - mu) / sigma
    query_z = (_vec(meta) - mu) / sigma
    dists = np.linalg.norm(stored_z - query_z, axis=1)
    order = np.argsort(dists)[:k]
    return [portfolio[i]["dataset"] for i in order]