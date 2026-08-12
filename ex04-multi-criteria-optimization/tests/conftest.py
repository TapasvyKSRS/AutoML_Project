"""
This file contains `fixtures`.
These are things that get injected directly into tests and are
rebuilt every "scope". In this case, everytime a function
requests one, the fixture will run again.
https://docs.pytest.org/en/6.2.x/fixture.html#what-fixtures-are
"""
import numpy as np
from pytest import fixture
from sklearn.datasets import load_wine

WINE_DATA = load_wine()

# 1, 2, 6 as features
# metric contains "malic acid", "ash", "flavanoids"


@fixture(scope="function")
def X() -> np.ndarray:
    return WINE_DATA["data"].copy()


@fixture(scope="function")
def X2d(X: np.ndarray) -> np.ndarray:
    return X[:, [1, 6]]


@fixture(scope="function")
def X3d(X: np.ndarray) -> np.ndarray:
    return X[:, [1, 2, 6]]
