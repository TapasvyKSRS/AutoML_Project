import numpy as np
from pytest import approx

from src.pareto import nDS, pareto


def test_pareto_2D(X2d: np.ndarray) -> None:
    pareto_front = pareto(X2d)
    selected_indices = list(np.argwhere(pareto_front).flatten())

    true_indices = [59, 113, 138, 141, 146, 165, 170, 171]
    assert selected_indices == true_indices


def test_pareto_3D(X3d: np.ndarray) -> None:
    pareto_front = pareto(X3d)
    selected_indices = list(np.argwhere(pareto_front).flatten())

    true_indices = [59, 76, 113, 138, 141, 145, 146, 165, 170, 171]
    assert selected_indices == true_indices


def test_NDS_2D(X2d: np.ndarray) -> None:
    # Test the non-dominating sorting method
    fronts = nDS(X2d)

    idxs = [59, 113, 138, 141, 146, 165, 170, 171]
    true_indices = X2d[idxs]

    # The first front should be the pareto front
    np.testing.assert_array_equal(fronts[0], true_indices)

    # Last is only a single point
    np.testing.assert_array_equal(fronts[-1], [[2.05, 5.08]])


def test_NDS_3D(X3d: np.ndarray) -> None:
    # Test the non-dominating sorting method in 3D
    fronts = nDS(X3d)

    idxs = [59, 76, 113, 138, 141, 145, 146, 165, 170, 171]
    true_indices = X3d[idxs]

    # The first front should be the pareto front
    np.testing.assert_array_equal(fronts[0], true_indices)

    # last is only a single point
    np.testing.assert_array_equal(fronts[-1], [[2.05, 3.23, 5.08]])
