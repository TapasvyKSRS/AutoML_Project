from __future__ import annotations

import numpy as np


def apriori(
    costs: np.ndarray,
    weights: list[float] | None = None,
    order: list[int] | None = None,
) -> int:
    """Implement the two example apriori methods presented in the lecture.

    Parameters
    ----------
    costs: np.ndarray, shape=(n_points, m_costs)
        The costs to consider

    weights: list[float] (m_costs,) | None = None
        Determines the weighting of the costs. Either this or `order` must be given

    order: list[int] (m_costs,) | None = None
        Priority of objectives for lexicographic minimization.
        Entries are objective indices, ordered from highest to lowest priority.
        Example: order=[2, 0, 1] means:
        1. Minimize costs[:, 2] first.
        2. If tied, minimize costs[:, 0].
        3. If still tied, minimize costs[:, 1].
        Typically this should be a permutation of 0..m_costs-1.
        Either this or weights must be given.

    Returns
    -------
    index: int
        Index of optimal element according to apriori method
    """

    if weights is not None and order is not None:
        raise ValueError("Can only specify either `weights` or `order`")

    index = -1
    if weights is not None:
        # TODO
        index = np.argmin(costs @ np.array(weights))

    elif order is not None:
        # TODO
        index = np.lexsort(costs[:, order[::-1]].T)[0]

    else:
        raise ValueError("Must specify either `weights` or `order`")

    return index
