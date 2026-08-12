from __future__ import annotations
import argparse
import logging
import random
import torch
import neps
from neps import PipelineSpace
import numpy as np
import traceback
from pathlib import Path
from typing import Any
from src.multicriterion_opt import eval_dt_config


def set_seed(seed: int):
    """
    Set the seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# TODO: define pipeline space, refer to `multicriterion_opt.py` and `eval_dt_config`
# We also recommend a visit to the NePS documentation: https://automl.github.io/neps/0.16.0/reference/neps_spaces/
class pipeline_space(PipelineSpace):
    max_leaf_nodes = neps.Integer(lower=2, upper=30)
    max_depth = neps.Integer(lower=2, upper=30)


def custom_target_scalarized(
    scalarization_weights: tuple[float | int, float | int] = (1, 1),
    method: str = "linear",
    **config,
) -> dict[str, Any]:
    """Combination of the two objectives: missclassification rate and runtime.

    Parameters:
    -----------
    scalarization_weights : tuple[float | int, float | int]
        Weights for the two objectives (error, runtime).

    method : str
        Scalarization method to combine objectives.

    config : dict
        Configuration dictionary with hyperparameters for the decision tree.

    Returns:
    --------
    dict[str, Any]
        A dictionary containing the scalarized loss, cost, and additional information.
        The scalarized loss is the combination of the two objectives.
    """

    max_leaf_nodes = config["max_leaf_nodes"]
    max_depth = config["max_depth"]
    k = 1  # number of folds kept constant for quicker runs, feel free to change, report accordingly

    error, runtime = eval_dt_config(max_leaf_nodes, max_depth, k)

    # TODO: Implement linear scalarization
    if method == "linear":
        scalarized_loss = scalarization_weights[0] * error + scalarization_weights[1] * runtime
    # TODO: Implement at least one non-linear scalarization method
    elif method == "chebyshev":
        scalarized_loss = max(
            scalarization_weights[0] * error,
            scalarization_weights[1] * runtime
        )
    elif method == "power":
        eps = 1e-8
        scalarized_loss = ((error+eps)**scalarization_weights[0] * (runtime+eps)**scalarization_weights[1])
        raise ValueError(f"Unknown scalarization method: {method}")

    return {
        "objective_to_minimize": scalarized_loss,
        "cost": runtime,
        "info_dict": {
            "error_rate": error,
            "runtime": runtime,
            "max_leaf_nodes": max_leaf_nodes,
            "max_depth": max_depth,
            "k": k,
        },
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NePS hyperparameter optimization for Decision Trees.")

    parser.add_argument(
        "--root_directory",
        type=str,
        default="./neps/",
        help="Output directory for NePS output.",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=30,
        help=(
            "Maximum number of evaluations to run. "
            "For 'grid_search', this must not exceed the number of available grid points (=16) "
            "in the pipeline space; otherwise NePS can stop with 'Grid search exhausted!'."
            )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Chosen seed for the run to ensure reproducibility."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        help=(
            "The search strategy/algorithm determining what configurations to evaluate next. "
            "Supported options include 'grid_search', 'random_search', 'bayesian_optimization', etc."
            "Refer to NePS documentation at "
            "[https://automl.github.io/neps/0.16.0/reference/search_algorithms/landing_page_algo/] "
            "for the full list of supported optimizers."
            )
    )
    parser.add_argument(
        "--method",
        type=str,
        default="linear",
        help="Scalarization method to combine the error rate and runtime objectives into a single loss. "
             "Options: 'linear' etc. (you can implement more linear/non-linear methods as part of the exercise)"
    )
    parser.add_argument(
        "--scalarization_weights",
        type=float,
        nargs="+",
        required=True,
        help=(
            "The list of scalarization weights in the order of "
            "metrics returned by eval_dt_config() (e.g. error, runtime)."
            )
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="To print NePS logs or not.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Create a function optimize using the provided scalarization weights
    def target_function(**config):
        try:
            return custom_target_scalarized(
                scalarization_weights=args.scalarization_weights,
                method=args.method,
                **config,
            )
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            raise e

    # Running NePS
    set_seed(args.seed)
    root_directory = Path(args.root_directory) / f"optimizer={args.optimizer}" /\
        f"method={args.method}" / f"seed={args.seed}"

    neps.run(
        evaluate_pipeline=target_function,
        pipeline_space=pipeline_space(),
        root_directory=root_directory,
        evaluations_to_spend=args.max_evals,
        optimizer=args.optimizer,
        overwrite_root_directory=True,  # choose as per your preference/need
    )
