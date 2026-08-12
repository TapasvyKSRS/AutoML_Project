"""
Multi-seed benchmark: the central scientific artifact for the poster.

For every dataset and every method, we run multiple seeds and report
mean +/- std of the test R^2. Methods compared:

    * defaults       : each model family with default hyperparameters (no HPO)
    * random_search  : uniform random config sampling (dumb search)
    * pipeline       : our full AutoML pipeline (built-in multi-fidelity HPO)
    * pipeline_smac  : our pipeline with the real SMAC BO backend (optional)

This directly serves two exam requirements:
    - "compare against simple baselines (random search, default hyperparameters)"
    - "base decisions on adequate repetitions and variance estimates"

Usage:
    python benchmark.py --datasets ds_linear ds_nonlinear --seeds 1 2 3 \
        --methods defaults random_search pipeline --n_iters 30 --out results.csv
"""
from __future__ import annotations

import logging, warnings
for _n in ("smac", "tabpfn", "distributed", "dask", "matplotlib"):
    logging.getLogger(_n).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from automl.data import Dataset
from automl.automl import AutoML
from automl.preprocessing import build_preprocessor
from automl.algo_selection import screen_algorithms
from automl.baselines import default_baseline, random_search
from automl.models import build_model

FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"


def _prep_split(X, y, seed):
    """Preprocess and split, returning arrays ready for baselines."""
    Xtr, Xval, ytr, yval = train_test_split(
        X, np.asarray(y).ravel(), test_size=0.2, random_state=seed
    )
    prep = build_preprocessor(X)
    Xtr_p = prep.fit_transform(Xtr)
    Xval_p = prep.transform(Xval)
    return prep, Xtr_p, ytr, Xval_p, yval


def run_baseline_method(method, X_train, y_train, X_test, y_test,
                        n_iters, cv_folds, seed):
    """Run a baseline (defaults / random_search) end-to-end and return test R^2."""
    prep, Xtr_p, ytr, Xval_p, yval = _prep_split(X_train, y_train, seed)
    top_models, _ = screen_algorithms(Xtr_p, ytr, Xval_p, yval,
                                      subset_fraction=0.3, top_k=2, seed=seed)

    if method == "defaults":
        res = default_baseline(top_models, Xtr_p, ytr, Xval_p, yval,
                               cv_folds=cv_folds, seed=seed)
    elif method == "random_search":
        res = random_search(top_models, Xtr_p, ytr, Xval_p, yval,
                           n_iters=n_iters, cv_folds=cv_folds, seed=seed)
    else:
        raise ValueError(method)

    # refit best config on all training data, score on test
    final_prep = build_preprocessor(X_train)
    X_full = final_prep.fit_transform(X_train)
    X_te = final_prep.transform(X_test)
    model = build_model(res.best_model_name, res.best_config, seed=seed)
    model.fit(X_full, np.asarray(y_train).ravel())
    return r2_score(y_test, model.predict(X_te))


def run_pipeline_method(backend, X_train, y_train, X_test, y_test,
                        n_iters, cv_folds, seed):
    """Run the full AutoML pipeline and return test R^2."""
    automl = AutoML(seed=seed, n_iters=n_iters, cv_folds=cv_folds,
                    backend=backend)
    automl.fit(X_train, y_train)
    return r2_score(y_test, automl.predict(X_test))


def main(args):
    rows = []
    for task in args.datasets:
        for method in args.methods:
            scores = []
            # Iterate official outer FOLDS as well as seeds: multiple seeds
            # on one fold do not measure variation across data splits.
            for fold in args.folds:
              for seed in args.seeds:
                ds = Dataset.load(datadir=args.datadir, task=task, fold=fold)
                if ds.y_test is None:
                    print(f"[skip] {task} has no y_test")
                    break
                t0 = time.time()
                if method in ("defaults", "random_search"):
                    r2 = run_baseline_method(
                        method, ds.X_train, ds.y_train, ds.X_test, ds.y_test,
                        args.n_iters, args.cv_folds, seed,
                    )
                elif method == "pipeline":
                    r2 = run_pipeline_method(
                        "builtin", ds.X_train, ds.y_train, ds.X_test, ds.y_test,
                        args.n_iters, args.cv_folds, seed,
                    )
                elif method == "pipeline_bohb":
                    r2 = run_pipeline_method(
                        "bohb", ds.X_train, ds.y_train, ds.X_test, ds.y_test,
                        args.n_iters, args.cv_folds, seed,
                    )
                elif method == "pipeline_smac":
                    r2 = run_pipeline_method(
                        "smac", ds.X_train, ds.y_train, ds.X_test, ds.y_test,
                        args.n_iters, args.cv_folds, seed,
                    )
                else:
                    raise ValueError(method)
                elapsed = time.time() - t0
                scores.append(r2)
                print(f"[{task:20s}] {method:14s} seed={seed} "
                      f"R2={r2:.4f} ({elapsed:.1f}s)")

            if scores:
                rows.append({
                    "dataset": task,
                    "method": method,
                    "mean_r2": np.mean(scores),
                    "std_r2": np.std(scores),
                    "n_runs": len(scores),
                    "folds": ";".join(str(f) for f in args.folds),
                    "scores": ";".join(f"{s:.4f}" for s in scores),
                })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n==== SUMMARY (mean +/- std test R^2) ====")
    for task in args.datasets:
        sub = df[df.dataset == task]
        if sub.empty:
            continue
        print(f"\n{task}:")
        for _, r in sub.iterrows():
            print(f"  {r['method']:14s}: {r['mean_r2']:.4f} +/- {r['std_r2']:.4f}")
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+",
                        default=["defaults", "random_search", "pipeline"],
                        choices=["defaults", "random_search", "pipeline",
                                 "pipeline_smac", "pipeline_bohb"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--folds", nargs="+", type=int, default=[1],
                        help="Official outer folds to evaluate on (1..10).")
    parser.add_argument("--n_iters", type=int, default=30)
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results.csv"))
    parser.add_argument("--datadir", type=Path, default=DATADIR)
    args = parser.parse_args()
    main(args)