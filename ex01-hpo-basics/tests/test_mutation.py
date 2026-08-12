from typing import Any, Optional, Tuple, Union

import numpy as np
from pytest import approx, mark, raises

from src.evolution import Member, Mutation, Recombination
from src.target_function import ackley


def make_member(
    arr: Any,
    method: Mutation,
    bounds: Tuple[float, float] = (-30, 30),
    sigma: Optional[float] = None,
) -> Member:
    return Member(
        initial_x=np.asarray(arr),
        target_function=ackley,
        bounds=bounds,
        mutation=method,
        recombination=Recombination.NONE,
        sigma=sigma,
    )


def test_none() -> None:
    """
    Expects
    -------
    * The coordinates and fitness should be equal between parent and child
        when mutation does nothing
    """
    parent = make_member([0], Mutation.NONE)
    child = parent.mutate()

    assert parent.fitness == approx(child.fitness)
    np.testing.assert_equal(parent.x_coordinate, child.x_coordinate)


def test_uniform() -> None:
    """
    Expects
    -------
    * Mutating uniformly `n` times should produce `n` unique values, with some tolerance
    """
    member = make_member([0.0, 0.0], Mutation.UNIFORM)
    n_mutations = 1_000
    tolerance = 3

    values = [member.mutate().x_coordinate for _ in range(n_mutations)]
    unique = np.unique(values, axis=0)

    assert len(unique) + tolerance >= len(values)


@mark.parametrize("mutation, sigma", [(Mutation.NONE, None), (Mutation.UNIFORM, None), (Mutation.GAUSSIAN, 1.0)])
def test_borders(mutation: Mutation, sigma: Union[None, float]) -> None:
    """
    Parameters
    ----------
    mutation: Mutation
        The mutation method

    sigma: Union[None, float]
        The sigma to use, if relevant

    Expects
    -------
    * A mutation should never go outside the bounds defined
    """
    n_mutations = 1_000

    member = make_member([-1, -1], method=mutation, bounds=(-1, 1), sigma=sigma)
    offspring = np.vstack([member.mutate().x_coordinate for _ in range(n_mutations)])

    lower, upper = member._bounds
    assert np.all((lower <= offspring) & (offspring <= upper))


def test_gauss_raises_exception_with_no_sigma() -> None:
    """
    Expects
    -------
    * A member with a Guassian mutation strategy with no sigma set should raise an Error
    """
    member = make_member([0], Mutation.GAUSSIAN, sigma=None)
    with raises(ValueError):
        member.mutate()


@mark.parametrize("x, sigma", [(np.array([0, 0]), 1.0), (np.array([10, -5]), 6.0)])
def test_gauss(x: np.ndarray, sigma: float) -> None:
    """
    Parameters
    ----------
    x : np.ndarray
        The starting coordinate of the Member

    sigma : float
        The sigma value to use

    Expects
    -------
    * The children of a member with a guassian mutation should have a mean centered on
    the original member and the correct stdev
    """
    n_mutations = 1_000
    member = make_member(x, Mutation.GAUSSIAN, sigma=sigma)

    offspring = np.vstack([member.mutate().x_coordinate for _ in range(n_mutations)])

    mean = np.mean(offspring, axis=0)
    stdev = np.std(offspring, axis=0)

    expected_stdev = np.array([sigma, sigma])

    np.testing.assert_allclose(x, mean, rtol=0.1, atol=0.3)
    np.testing.assert_allclose(expected_stdev, stdev, rtol=0.1, atol=0.3)
