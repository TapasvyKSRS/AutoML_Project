from typing import Callable

import numpy as np
import pytest

from src.main import EI, LCB, f, gp_model, run_grid_search, run_random_search, manual_fit_gaussian_process


def test_lower_confidence_bound():
    """Test lower confidence bound."""
    np.random.seed(0)
    x = np.linspace(-15, 10, 10).reshape(-1, 1)
    y = [f(i) for i in x]
    gp = gp_model()
    gp.fit(x, y)  # fit the model

    ucb = [LCB(i, gp, min(y), 1) for i in x]
    expected = np.array(
        [
            15.93409352,
            21.17581312,
            10.4529669,
            4.59195362,
            3.28877953,
            1.21268212,
            0.23226244,
            1.28790492,
            3.6071361,
            10.55681337,
        ]
    )
    expected = expected.reshape(-1, 1)
    np.testing.assert_allclose(ucb, expected, atol=1e-3)


def test_expected_improvement():
    """Test that expected improvement yields the expected result"""
    np.random.seed(0)
    x = np.linspace(-15, 10, 10).reshape(-1, 1)
    y = [f(i) for i in x]

    gp = gp_model()
    gp.fit(x, y)  # fit the model

    ei = [EI(i, gp, min(y)) for i in x]

    expected = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.53433968e-51,
            2.64141079e-02,
            8.07334916e-59,
            0.0,
            0.0,
        ]
    )
    expected = expected.reshape(-1, 1)
    np.testing.assert_allclose(ei, expected, atol=1e-4)

def test_manual_fit_gaussian_process():
    """Test manual fit gaussian process"""
    np.random.seed(0)
    x = np.asarray([0, 0.5, 1])
    y = np.asarray([1, 0, 0.5])

    x_star = np.linspace(-1, 2, 2)

    mu, var = manual_fit_gaussian_process(x, y, x_star)
    print(mu, var)

    mu_expected = np.array(
        [
            2.88766645,
            2.28889632
        ]
    )


    var_expected = np.array(
        [
            [ 
              0.3210979,
              -0.10351767
              ],

            [
              -0.10351767,
              0.3210979 
              ]
          ]
    )

    np.testing.assert_allclose(mu, mu_expected, atol=1e-3)
    np.testing.assert_allclose(var, var_expected, atol=1e-3)

@pytest.mark.parametrize("search_method", [run_random_search, run_grid_search])
def test_manual_search_methods(search_method: Callable[[int], np.ndarray]):
    """Test manual search method

    Expects
    -------
    * All samples found should be in [0, 30]
    """
    n_samples = 100000
    samples = search_method(n_samples)

    assert all(0 <= sample <= 30 for sample in samples)
