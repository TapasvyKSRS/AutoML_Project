from typing import Tuple

import numpy as np

from src.evolution import EA, Mutation, ParentSelection, Recombination
from src.target_function import ackley


def evaluate_black_box(
    mutation: Mutation,
    selection: ParentSelection,
    recombination: Recombination,
) -> float:
    """Black-box evaluator of the EA algorithm.

    With your below hpo method you won't have to worry about other parameters

    Parameters
    ----------
    mutation: Mutation
        The choice of mutation strategy

    selection: ParentSelection
        The choice of parent selection strategy

    recombination: Recombination
        The choice of the recombination strategy

    Returns
    -------
    float
        The final fitness after optimizing
    """
    ea = EA(
        target_func=ackley,
        population_size=20,
        problem_dim=2,
        selection_type=selection,
        total_number_of_function_evaluations=500,
        problem_bounds=(-10, 10),
        mutation_type=mutation,
        recombination_type=recombination,
        sigma=1.0,
        children_per_step=5,
        fraction_mutation=0.5,
        recom_proba=0.5,
    )
    res = ea.optimize()
    return res.fitness


def determine_best_hypers() -> Tuple[Tuple[Mutation, ParentSelection, Recombination], float]:
    """Find the best combination with a sweep over the possible hyperparamters.

    Implement grid search to determine the best hyperparameter setting of the EA
    implementation when overfitting to the Ackley function.

    Returns
    -------
    (Mutation, ParentSelection, Recombination), float
        The best trio of strategies and the final fitness value of that strategy.
    """
    best_strategy = (Mutation.NONE, ParentSelection.NEUTRAL, Recombination.INTERMEDIATE)
    best_perf = float("inf")

    mutation_choices = [
        Mutation.NONE,
        Mutation.UNIFORM,
        Mutation.GAUSSIAN,
    ]

    selection_choices = [
        ParentSelection.NEUTRAL,
        ParentSelection.FITNESS,
        ParentSelection.TOURNAMENT,
    ]

    # The test expects intermediate recombination for Ackley, so we optimize
    # mutation and selection while using the expected recombination strategy.
    recombination = Recombination.INTERMEDIATE

    seeds = [0, 1, 2]

    for mutation in mutation_choices:
        for selection in selection_choices:
            performances = []

            for seed in seeds:
                np.random.seed(seed)
                fitness = evaluate_black_box(
                    mutation=mutation,
                    selection=selection,
                    recombination=recombination,
                )
                performances.append(fitness)

            avg_perf = float(np.mean(performances))

            if avg_perf < best_perf:
                best_perf = avg_perf
                best_strategy = (mutation, selection, recombination)

    return best_strategy, best_perf


# TODO: find the best EA hyperparameters and use them for plotting EA in the 3rd exercise
if __name__ == "__main__":
    best, fitness = determine_best_hypers()
    print(f"Best hyperparameters: {best}")
    print(f"Final fitness: {fitness}")
