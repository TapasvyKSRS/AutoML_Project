from __future__ import annotations

import numpy as np
from tqdm import tqdm

from src.fcnet_benchmark import FCNetBenchmark
from src.problem import Problem
from src.successive_halving import successive_halving
from src.utils import plot_grey_box_optimization


# Hint 1: use pseudocode from https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/18-ICML-BOHB.pdf
def hyperband(
    problem: Problem,
    min_budget_per_model: int,
    max_budget_per_model: int,
    eta: float,
    random_seed: int | None = None,
) -> list:
    """The hyperband algorithm

    Parameters
    ----------
    problem : Problem
        A problem instance to run on

    min_budget_per_model : int
        The minimum budget per model

    max_budget_per_model : int
        The maximum budget per model

    eta : float
        The eta float parameter. The budget is multiplied by eta at each iteration

    random_seed : int | None = None
        The random seed to use

    Returns
    -------
    list[dict]
        A list of dictionaries with the config information
    """
    # TODO: Compute s_max
    # Hint 2: the budget starts at min_budget_per_model,
    # is multiplied by eta at each iteration,
    # and should be equal to max_budget_per_model at the end
    s_max = int(np.floor(np.log(max_budget_per_model / min_budget_per_model) / np.log(eta)))

    configs_dicts = []

    iterations = reversed(range(s_max + 1))

    # tqdm gives us a nice progress bar
    for s in tqdm(iterations, desc="Hyperband iter"):
        # TODO: Compute the number of models to evaluate in the HB iteration
        n_models = int(np.ceil(((s_max + 1) / (s + 1)) * eta**s))

        # TODO: Compute the min budget per model in the current HB iteration
        # According to pseudocode line 4: initial budget = eta^(-s) * b_max
        min_budget_per_model_iter = int(eta**(-s) * max_budget_per_model)

        configs_dict = successive_halving(
            problem=problem,
            n_models=n_models,
            min_budget_per_model=min_budget_per_model_iter,
            max_budget_per_model=max_budget_per_model,
            eta=eta,
            random_seed=random_seed,
        )
        configs_dicts.append(configs_dict)

    return configs_dicts


if __name__ == "__main__":
    problem = FCNetBenchmark(name="protein_structures")
    configs_dicts = hyperband(
        problem=problem,
        eta=2,
        random_seed=0,
        max_budget_per_model=100,
        min_budget_per_model=2,
    )
    # TODO: Plot Hyperband results
    plot_grey_box_optimization(configs_dicts, min_budget_per_model=2, kind="hyperband")
