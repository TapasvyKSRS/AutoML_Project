import numpy as np


def ackley(coordinate: np.ndarray) -> float:
    """n-dimensional Ackley function.

    Note: Bounded by -30 <= coordinate[i] <= 30.

    Parameters
    ----------
    coordinate: np.ndarray (dtype: float)
        The array to compute the function on

    Returns
    -------
    float
        The value of the ackley function on the coordinate
    """
    lower, upper = (-30, 30)
    if not np.all((lower <= coordinate) & (coordinate <= upper)):
        raise ValueError(f'Coordinates have to be in [{lower}, {upper}], got: \n{coordinate}')

    first_sum = 0.0
    second_sum = 0.0
    for c in coordinate:
        first_sum += c ** 2.0
        second_sum += np.cos(2.0 * np.pi * c)

    n = float(len(coordinate))
    return -20.0 * np.exp(-0.2 * np.sqrt(first_sum / n)) - np.exp(second_sum / n) + 20 + np.e
