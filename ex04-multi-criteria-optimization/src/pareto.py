from __future__ import annotations

import numpy as np


def pareto(costs: np.ndarray) -> np.ndarray:
    """Find the pareto-optimal points.

    Parameters
    ----------
    costs: np.ndarray (n_points, m_costs)

    Returns
    -------
    np.ndarray (n_points)
        Boolean array indicating if point is on pareto front or not.
    """
    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)

    # TODO
    # -------------
    # Stepwise eliminate all elements that are dominated by any other point
    for i in range(costs.shape[0]):
        if is_pareto[i]:
            is_dominated = np.all(costs[i] <= costs, axis=1) & np.any(costs[i] < costs, axis=1)
            is_dominated[i] = False
            is_pareto[is_dominated] = False

    return is_pareto


def nDS(costs: np.ndarray) -> list[np.ndarray]:
    """Non-dominated sorting method.

    Parameters
    ----------
    costs: np.ndarray (n_points, m_costs)

    Returns
    -------
    list[np.ndarray]
        A list where each element is an array representing a front.
    """
    # TODO
    # Stepwise compute the pareto front without all prior dominating points
    fronts = []
    remaining_costs = costs.copy()
    while remaining_costs.size > 0:
        pareto_front = pareto(remaining_costs)
        fronts.append(remaining_costs[pareto_front])
        remaining_costs = remaining_costs[~pareto_front]

    return fronts


def construct_polygon(pareto_front: np.ndarray, ref: tuple[float, float]) -> np.ndarray:
    """Construct a polygon for the Pareto front and reference point.

    Args:
        pareto_front (np.ndarray): The Pareto front points.
        ref: tuple[float, float]: The reference point for hypervolume calculation.

    Returns:
        np.ndarray: The polygon points.
    """
    # Sort front to get "outline" of polygon (don't forget to add the reference point to that outline)
    sorted_front = np.sort(
        pareto_front.view([("", pareto_front.dtype)] * pareto_front.shape[1]), axis=0
    ).view(np.float64)

    polygon = [ref]
    for x in sorted_front:
        elem_at = len(polygon) - 1
        # add intersection points by keeping the x constant
        polygon.append((x[0], polygon[elem_at][1]))
        polygon.append(x)

    polygon.append((polygon[0][0], polygon[-1][1]))
    return np.array(polygon)


def computeHV2D(front: np.ndarray, ref: tuple[float, float]) -> float:
    """Compute the Hypervolume for the pareto front (only implemented for 2D!).

    Parameters
    ----------
    front : np.ndarray (n_points, m_cost_values)
        Front for which to compute the volume
    ref : tuple[float, float]
        Coordinates of the reference point
    Returns
    -------
    float
        Hypervolume of the polygon spanned by all points in the front
        + the reference point
    """
    if front.ndim == 1:
        front = front.reshape(1, -1)

    # (https://en.wikipedia.org/wiki/Shoelace_formula)
    def shoelace(x_y):  # taken from https://stackoverflow.com/a/58515054
        x_y = np.array(x_y)
        x_y = x_y.reshape(-1, 2)

        x = x_y[:, 0]
        y = x_y[:, 1]

        S1 = np.sum(x * np.roll(y, -1))
        S2 = np.sum(y * np.roll(x, -1))

        area = 0.5 * np.absolute(S1 - S2)

        return area

    return shoelace(construct_polygon(front, ref))


if __name__ == "__main__":
    # We prepared some plotting code for you to check your pareto front implementation and the non-dominating sorting
    import seaborn as sb
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_wine

    sb.set_style("darkgrid")

    wine = load_wine()
    X = wine["data"]  # 1, 2, 6 as features
    # metric contains "malic acid", "ash", "flavanoids"
    costs2D = X[:, [1, 2]]
    costs3D = X[:, [1, 2, 6]]

    plt.scatter(costs2D[:, 0], costs2D[:, 1])
    pareto_front = costs2D[pareto(costs2D)]

    # sort for plotting
    pareto_front = np.sort(
        pareto_front.view([("", pareto_front.dtype)] * pareto_front.shape[1]),
        order=["f1"],  # order by first element then second and so on
        axis=0,
    ).view(float)
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], marker=".", c="orange")
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], marker=".", c="orange")
    plt.title("Pareto Front on partial Wine Dataset")
    plt.xlabel("malic acid")
    plt.ylabel("ash")
    plt.show()

    # metric contains malic acid and flavanoids
    costs2D = X[:, [1, 6]]

    plt.scatter(costs2D[:, 0], costs2D[:, 1])

    fronts = nDS(costs2D)
    for pareto_front in fronts:
        # sort for plotting
        pareto_front = np.sort(
            pareto_front.view([("", pareto_front.dtype)] * pareto_front.shape[1]),
            order=["f1"],  # order by first element then second and so on
            axis=0,
        ).view(float)
        plt.plot(pareto_front[:, 0], pareto_front[:, 1], marker=".")
    plt.title("Pareto Front on partial Wine Dataset")
    plt.xlabel("malic acid")
    plt.ylabel("flavanoids")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs = costs3D[:, 0]
    ys = costs3D[:, 1]
    zs = costs3D[:, 2]
    pareto_front = pareto(costs3D)

    ax.scatter(xs, ys, zs)
    ax.scatter(
        xs[pareto_front],
        ys[pareto_front],
        zs[pareto_front],
        c="orange",
        marker="X",
        alpha=1,
    )
    ax.set_xlabel("malic acid")
    ax.set_ylabel("ash")
    ax.set_zlabel("flavanoids")
    ax.view_init(10, -15)
    plt.show()
