import numpy as np

from src.apriori import apriori


def test_weighted_sum(X2d: np.ndarray) -> None:
    a = apriori(X2d, weights=[0, 1])
    assert a == np.argmin(X2d[:, 1])

    b = apriori(X2d, weights=[1, 0])
    assert b == np.argmin(X2d[:, 0])

    assert a != b


def test_weighted_sum_full(X: np.ndarray) -> None:
    a = apriori(X, weights=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    assert a == np.argmin(X[:, 5])

    b = apriori(X, weights=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    assert b == np.argmin(X[:, 8])

    c = apriori(X, weights=[0.1, 0.1, 0.2, 0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0])
    assert c == 59

    assert a != b
    assert a != c


def test_lexicographical(X2d: np.ndarray) -> None:
    a = apriori(X2d, order=[0, 1])
    assert a == np.argmin(X2d[:, 0])

    b = apriori(X2d, order=[1, 0])
    assert b == np.argmin(X2d[:, 1])

    assert a != b


def test_lexicographical_full(X: np.ndarray) -> None:
    a = apriori(X, order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert a == np.argmin(X[:, 0])

    b = apriori(X, order=[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    assert b == np.argmin(X[:, -1])
    assert a != b

    c = apriori(X, order=[7, 6, 5, 4, 12, 11, 10, 9, 8, 3, 2, 1, 0])
    assert c == np.argmin(X[:, 7])
    assert a != c
