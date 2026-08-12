from typing import Any

import numpy as np
from pytest import approx

from src.evolution import EA, Member, Mutation, ParentSelection, Recombination
from src.target_function import ackley


def make_member(x: Any) -> Member:
    if isinstance(x, (float, int)):
        x = [x]
    return Member(np.asarray(x), ackley, (-30, 30), Mutation.NONE, Recombination.NONE)


def test_tournament() -> None:
    """
    Expects
    -------
    * The optimal candidate should be chosen everytime with tournament selection
    and never be in the selected parents.
    """
    # With the fake population we should never choose member 1
    population = [make_member(x=0), make_member(x=30)]
    ea = EA(
        target_func=ackley,
        population_size=len(population),
        problem_dim=1,
        selection_type=ParentSelection.TOURNAMENT,
        total_number_of_function_evaluations=100,
        children_per_step=1,
    )
    ea.population = population

    parent_ids = ea.select_parents()
    assert parent_ids == [0]

    # Test that we also rather choose 0 multiple times than to choose 1 even once
    population = [make_member(x=0), make_member(x=30)]
    ea = EA(
        target_func=ackley,
        population_size=len(population),
        problem_dim=1,
        selection_type=ParentSelection.TOURNAMENT,
        total_number_of_function_evaluations=100,
        children_per_step=2,
    )
    ea.population = population

    parent_ids = np.array([ea.select_parents() for _ in range(100)])  # type: ignore
    assert 1 not in parent_ids


def test_fitness():
    """
    Expects
    -------
    * The member with the worst value of x (30) should not be included in selected
    * The count of how often they are selected should be ordered by their fitness,
        from the fitteset at (x=0) to least fit (x=5)
    """
    # With the fake population we should never choose member 1
    population = [make_member(x=x) for x in (0, 0.5, 2, 5, 30)]
    ea = EA(
        ackley,
        population_size=len(population),
        problem_dim=1,
        selection_type=ParentSelection.FITNESS,
        total_number_of_function_evaluations=100,
        children_per_step=1,
    )

    ea.population = population

    uniques, counts = np.unique([ea.select_parents() for _ in range(1_000)], return_counts=True)
    print(uniques, counts)

    # the others are proportionally sampled according to their fitness
    assert counts[0] > counts[1] > counts[2] > counts[3], f"Counts {counts}"


def test_neutral():
    """
    Expects
    -------
    * Each member should be included at least once, regardless of their fitness
    * They should each be included in roughly quantities
    """
    population = [make_member(x=x) for x in (0, 0.5, 2, 5, 30)]
    ea = EA(
        target_func=ackley,
        population_size=len(population),
        problem_dim=1,
        selection_type=ParentSelection.NEUTRAL,
        total_number_of_function_evaluations=100,
        children_per_step=1,
    )

    uniques, counts = np.unique([ea.select_parents() for _ in range(5_000)], return_counts=True)
    print(uniques, counts)

    assert [0, 1, 2, 3, 4] == uniques.tolist()

    # zero-sum-game -> mean is easy to determine
    assert np.mean(counts) == approx(1000, abs=1)
