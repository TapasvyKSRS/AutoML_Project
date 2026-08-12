from typing import Any, Optional

import numpy as np
from pytest import raises

from src.evolution import Member, Mutation, Recombination
from src.target_function import ackley


def make_member(arr: Any, method: Recombination, recom_prob: Optional[float] = None) -> Member:
    return Member(
        initial_x=np.asarray(arr),
        target_function=ackley,
        bounds=(-30, 30),
        mutation=Mutation.NONE,
        recombination=method,
        recom_prob=recom_prob,
    )


def test_none() -> None:
    """
    Expects
    -------
    * With no recombination method, their fitness and x_coordinate should be the same
    """
    parent = make_member([0], Recombination.NONE)
    child = make_member([0], Recombination.NONE)

    assert child.fitness == parent.fitness
    np.testing.assert_equal(child.x_coordinate, parent.x_coordinate)


def test_uniform() -> None:
    """
    Expects
    -------
    * Combining both members should give all 4 possible combinations
    """
    a = make_member([0, 0], Recombination.UNIFORM, recom_prob=0.5)
    b = make_member([1, 1], Recombination.UNIFORM, recom_prob=0.5)

    unique, counts = np.unique(
        [a.recombine(b).x_coordinate for _ in range(1_000)],
        axis=0,
        return_counts=True,
    )
    assert sum(counts) == 1_000

    # we have exactly 4 possible combinations
    assert len(unique) == 4

    # All are equally likely -> roughly 250 occurances per outcome
    assert np.allclose([250, 250, 250, 250], counts, rtol=10, atol=1)


def test_uniform_raises_without_recom_prob() -> None:
    """
    Expects
    -------
    * Combining with with no recom_prob should raise an Error
    """
    a = make_member([0, 0], Recombination.UNIFORM, recom_prob=None)
    b = make_member([1, 1], Recombination.UNIFORM, recom_prob=0.5)

    with raises(ValueError):
        a.recombine(b)


def test_intermediate() -> None:
    """
    Expects
    -------
    * The children of two intermediate nodes should be the mean of their values
    """
    a = make_member([0], Recombination.INTERMEDIATE)
    b = make_member([0], Recombination.INTERMEDIATE)

    offspring = a.recombine(b)

    expected = np.mean([a.x_coordinate, b.x_coordinate], axis=0)

    np.testing.assert_allclose(offspring.x_coordinate, expected, rtol=0.1, atol=0.3)
