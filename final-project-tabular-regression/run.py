"""An example run file which loads in a dataset from its files
and logs the R^2 score on the test set.

For the practice datasets you are given access to y_test; however,
for the exam dataset, you will not have access to these labels.
You will need to output your predictions for X_test to a file,
which we will grade using GitHub Classroom.
"""
from __future__ import annotations

import logging, warnings
for _n in ("smac", "tabpfn", "distributed", "dask", "matplotlib"):
    logging.getLogger(_n).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
from automl.data import Dataset
from automl.automl import AutoML
import argparse

import logging

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"


def main(
    task: str,
    fold: int,
    output_path: Path,
    seed: int,
    datadir: Path,
    n_iters: int = 50,
    cv_folds: int = 3,
    top_k: int = 3,
    backend: str = "builtin",
    prefer: str = "accuracy",
    portfolio_path: str | None = None,
):
    import time
    _t_start = time.perf_counter()

    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)

    logger.info("Fitting AutoML")

    # You do not need to follow this setup or API it's merely here to provide
    # an example of how your automl system could be used.
    # As a general rule of thumb, you should **never** pass in any
    # test data to your AutoML solution other than to generate predictions.
    automl = AutoML(
        seed=seed,
        n_iters=n_iters,
        cv_folds=cv_folds,
        top_k=top_k,
        backend=backend,
        prefer=prefer,
        portfolio_path=portfolio_path,
    )
    automl.fit(dataset.X_train, dataset.y_train)
    test_preds: np.ndarray = automl.predict(dataset.X_test)

    elapsed = time.perf_counter() - _t_start
    logger.info(f"Total wall-clock time: {elapsed:.1f}s ({elapsed/3600:.3f} h) -- {elapsed/86400*100:.2f}% of 24h budget")

    # Write the predictions of X_test to disk
    # This will be used by github classrooms to get a performance
    # on the test set.
    # --- validate predictions before writing (submission safety) ---
    test_preds = np.asarray(test_preds).ravel()
    n_expected = len(dataset.X_test)
    if test_preds.shape[0] != n_expected:
        raise ValueError(
            f"Prediction length {test_preds.shape[0]} != len(X_test) {n_expected}")
    if not np.all(np.isfinite(test_preds)):
        raise ValueError("Predictions contain NaN or infinite values")

    logger.info("Writing predictions to disk")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    # save resolved configuration next to predictions (reproducibility)
    import json
    with output_path.with_suffix(".config.json").open("w") as f:
        json.dump({"task": task, "fold": fold, "seed": seed,
                   "n_iters": n_iters, "cv_folds": cv_folds,
                   "backend": backend, "prefer": prefer,
                   "portfolio_path": portfolio_path,
                   "wall_clock_seconds": round(elapsed, 2),
                   "summary": automl.summary_}, f, indent=2, default=str)

    if dataset.y_test is not None:
        r2_test = r2_score(dataset.y_test, test_preds)
        logger.info(f"R^2 on test set: {r2_test}")
    else:
        # This is the setting for the exam dataset, you will not have access to y_test
        logger.info(f"No test labels (y_test) for task '{task}'")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The name of the task to run on.",
        choices=["bike_sharing_demand", "brazilian_houses", "superconductivity", "wine_quality", "yprop_4_1", "exam_dataset"]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/bike_sharing_demand/1/predictions.npy"),
        help=(
            "The path to save the predictions to."
            " Parent directories are created automatically."
        )
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help=(
            "The fold to run on."
            " You are free to also evaluate on other folds for your own analysis."
            " For the exam dataset, there exists only a single fold, fold 1."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using and randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        )
    )

    parser.add_argument(
        "--datadir",
        type=Path,
        default=DATADIR,
        help=(
            "The directory where the datasets are stored."
            " You should be able to mostly leave this as the default."
        )
    )
    parser.add_argument("--n_iters", type=int, default=50,
                        help="HPO budget (number of configurations).")
    parser.add_argument("--cv_folds", type=int, default=3,
                        help="Inner cross-validation folds used during HPO.")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Model families kept by screening for HPO.")
    parser.add_argument("--backend", type=str, default="builtin",
                        choices=["builtin", "smac", "bohb"],
                        help="HPO backend: builtin (multi-fidelity random search), smac (Bayesian optimisation), or bohb (BOHB).")
    parser.add_argument("--prefer", type=str, default="accuracy",
                        choices=["accuracy", "speed"],
                        help="Pareto-front selection preference.")
    parser.add_argument("--portfolio_path", type=str, default=None,
                        help="Meta-learning portfolio JSON for warm-starting.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Running task {args.task}"
        f"\n{args}"
    )

    main(
        task=args.task,
        fold=args.fold,
        output_path=args.output_path,
        datadir=args.datadir,
        seed=args.seed,
        n_iters=args.n_iters,
        cv_folds=args.cv_folds,
        top_k=args.top_k,
        backend=args.backend,
        prefer=args.prefer,
        portfolio_path=args.portfolio_path,
    )