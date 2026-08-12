from __future__ import annotations
from operator import inv
import numpy as np
import os
import pickle
import argparse
from typing import Any, Callable, Dict, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, Kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

rcParams.update({"figure.autolayout": True})
plt.style.use("seaborn-v0_8-whitegrid")


def kernel_matrix(x: Sequence, y: Sequence) -> np.ndarray:
    """Given two vectors, calculate the kernel matrix

    K_ij = k(x_i, y_j), where k is the squared exponential kernel

    Parameters
    ----------
    x, y : Sequence
        The vectors to calculate the kernel matrix from

    Returns
    -------
    np.ndarray
        The kernel matrix
    """
    return np.exp(-0.5 * np.subtract.outer(x, y)**2)


def manual_fit_gaussian_process(x: Sequence, y: Sequence, x_star: float) -> tuple[float, float]:
    """Fit a gaussian process on a sequence of observations (x_i, f_i).

    Parameters
    ----------
    x : Sequence
        The x values of the observations. Shape: (n,)

    y : Sequence
        The y values of the observations. Shape: (n,)

    x_star : float
        The x value to predict at

    Returns
    -------
    y_star : float
        The predicted mean. Calculated as: K_star.T @ inv(K) @ y
    sigma_star : float
        The predicted variance. Calculated as: K_star_star - K_star.T @ inv(K) @ K_star
    """

    K = kernel_matrix(x, x)
    K_star = [kernel_matrix(x_star, x_) for x_ in x]
    K_star_star = kernel_matrix(x_star, x_star)

    # TODO: Calculate the mean and variance of the prediction for x_star
    predicted_mean = np.transpose(K_star) @ np.linalg.solve(K, y)
    predicted_variance = K_star_star - np.transpose(K_star) @ np.linalg.solve(K, K_star)
    return predicted_mean, predicted_variance


def gp_model(
    kernel: Optional[Kernel] = None,
    alpha: float = 1.0e-4,
    normalize_y: bool = True,
    n_restarts_optimizer: int = 10,
) -> Pipeline:
    """Create a gp model used for BO."""
    if kernel is None:
        kernel = Matern(nu=2.5, length_scale_bounds=(1e-8, 1e5))
    model = GPR(
        kernel=kernel,
        alpha=alpha,
        normalize_y=normalize_y,
        n_restarts_optimizer=n_restarts_optimizer,
    )
    return Pipeline([["standardize", StandardScaler()], ["GP", model]])


def f(x: float | Sequence) -> float:
    """Calculate function that's used to minimize.

    Levy1D see https://www.sfu.ca/~ssurjano/levy.html.
    Global min value: 0.0

    Parameters
    ----------
    x: float | Sequence
        The value to calculate f at, can be a single number or an array like
        such as np.ndarry

    Returns
    -------
    float
        The functions value at x
    """
    if isinstance(x, float):
        x = [x]

    w0 = 1 + (x[0] - 1) / 4
    term1 = np.power(np.sin(np.pi * w0), 2)

    term2 = 0
    for i in range(len(x) - 1):
        wi = 1 + (x[i] - 1) / 4
        term2 += np.power(wi - 1, 2) * (1 + 10 * np.power(np.sin(wi * np.pi + 1), 2))

    wd = 1 + (x[-1] - 1) / 4
    term3 = np.power(wd - 1, 2)
    term3 *= 1 + np.power(np.sin(2 * np.pi * wd), 2)

    y = term1 + term2 + term3
    return y


def EI(x: float, model: Pipeline, eta: float) -> float:
    """Calculate Expected Improvement.

    Parameters
    ----------
    x : float
        The value to determine the acquisition value at

    model : Pipeline
        The model to predict with

    eta : float
        The best value seen so far

    Returns
    -------
    float
        The expected improment for a given model at x
    """
    # Required as models predict on a list of values normally, not just one
    x = np.array([x]).reshape([-1, 1])

    m, s = model.predict(x, return_std=True)

    # TODO: Implement Expected Improvement
    with np.errstate(divide='warn'):
        imp = eta - m
        Z = imp / s
        ei = imp * norm.cdf(Z) + s * norm.pdf(Z)
        ei[s == 0.0] = 0.0
    return ei.flatten()


def LCB(x: float, model: Pipeline, eta: float, alpha: float = 3.0) -> float:
    """Calculate Lower confidence bound.

    Parameters
    ----------
    x : float
        The value to determine the acquisition value at

    model : Pipeline
        The model to predict with

    eta : float (IGNORED)
        This variable is ignored but kept for consistency

    alpha : float = 3.0
        The additional value to add to the LCB

    Returns
    -------
    float
        The lower confidence bound for a given model at x
    """
    # Required as models predict on a list of values normally, not just one
    x = np.array([x]).reshape([-1, 1])

    m, s = model.predict(x, return_std=True)

    # TODO: Implement Lower Confidence Bound
    # return float(m[0] - alpha * s[0])
    result = m - alpha * s
    return result.flatten()


def run_random_search(max_iter: int, seed: int = 1) -> np.ndarray:
    """Random Search.

    Parameters
    ----------
    max_iter : int
        The max number of iterations to run for

    seed : int
        The seed to use for the randomness to keep experiments reproducible

    Returns
    -------
    np.ndarray
        The evaluated points
    """
    # TODO: Implement random search. Hint : the search space is the same as in the BO method.
    np.random.seed(seed)
    x = np.random.uniform(-15, 10, max_iter)
    return np.array([f(xi) for xi in x])


def run_grid_search(max_iter: int, grid_size: int | None = None) -> np.ndarray:
    """Grid Search.

    Parameters
    ----------
    max_iter : int
        The max number of iterations to run for

    grid_size : int | None = None
        The grid size to use. If this is specified, it may not produce less points than
        max_iter.

    Returns
    -------
    np.ndarray
        The evaluated points
    """
    # --- Grid Search means points are evenly spaced inside the interval
    # TODO: Implement grid search
    n = grid_size if grid_size is not None else max_iter
    x = np.linspace(-15, 10, n)
    return np.array([f(xi) for xi in x])


def run_bo(
    acquisition: Callable,
    min_or_max: str,
    max_iter: int,
    init: int = 25,
    randomize: bool = True,
    seed: int = 1,
    acq_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Run the BO loop.

    Parameters
    ----------
    acquisition : (x: float, model: Pipeline, eta: float, **func_params) -> float
        The acquisition function to use

    min_or_max: "min" | "max"
        Whether to minimize or maximise the acquisition function

    max_iter : int
        The max number of iterations to run for

    init : int = 25
        Number of points to build initial model with

    randomize: bool = True
        Whether to randomize initial points. If False, samples are lineraly sampled,
        otherwise the are sampled uniformly ad random

    seed : int = 1
        The seed to use for randomization

    acq_params: Optional[Dict[str, Any]] = None
        Any additional arguments to pass to the acquisition function

    Returns
    -------
    np.ndarray
        The evaluated points
    """
    # sample initial query points
    np.random.seed(seed)
    if acq_params is None:
        acq_params = {}

    sign = -1 if min_or_max == "max" else 1

    if randomize:
        x = np.random.uniform(-15, 10, init)
    else:
        x = np.linspace(-15, 10, init)

    # get corresponding response values
    y = list(map(f, x))

    for i in range(max_iter - init):  # BO loop
        # Feel free to adjust the hyperparameters
        gp = gp_model()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gp.fit(x.reshape(-1, 1), y)
        # Multi-start optimization makes acquisition search more robust.
        n_restarts = 5
        best_res = None

        for _ in range(n_restarts):
            opt_res = minimize(
                fun=lambda x: sign * acquisition(x=x, model=gp, eta=min(y), **acq_params),
                x0=[np.random.uniform(-15, 10)],
                bounds=[(-15, 10)],
                options={"maxfun": 30},
                method="L-BFGS-B",
            )
            if opt_res.success and np.isfinite(opt_res.fun):
                if best_res is None or opt_res.fun < best_res.fun:
                    best_res = opt_res

        if best_res is not None:
            x_new = float(best_res.x[0])
        else:
            print(f"[Warning] Optimization failed at iteration {i}. Using random fallback point.")
            x_new = float(np.random.uniform(-15, 10))

        x = np.append(x, x_new)
        y.append(f(x_new))

    return np.asarray(y)


def main(
    num_evals: int = 100,
    init_size: float = 0.25,
    repetitions: int = 5,
    randomize: bool = True,
    seed: int = 1,
    load_prior_results=False,
) -> None:
    np.random.seed(0)
    x = np.asarray([0, 0.5, 1])
    y = np.asarray([1, 0, 0.5])

    # Manually fit a Gaussian Process on
    # x, y using the manual_fit_gaussian_process function

    x_star_list = np.linspace(-1, 2, 100)
    predictions_gaussian = [manual_fit_gaussian_process(x, y, x_star) for x_star in x_star_list]
    y_star, sigma_star = np.transpose(predictions_gaussian)

    # Plot the mean and variance of the fitted Gaussian Process

    plt.plot(x_star_list, y_star, 'b-')
    plt.plot(x, y, 'ro')
    plt.fill_between(x_star_list, y_star - 1 * np.sqrt(np.maximum(sigma_star, 0)),
                     y_star + 1 * np.sqrt(np.maximum(sigma_star, 0)), alpha=0.2)
    plt.tight_layout()
    plt.show()

    init = int(init_size * num_evals)

    # Do some plots
    rng = np.random.RandomState(2)

    # fit the model to some intial points
    gp = gp_model()
    x = rng.uniform(-15, 10, 10)
    y = [f(i) for i in x]
    gp.fit(x.reshape(-1, 1), y)

    x_axis = np.linspace(-15, 10, 500)

    # Have each method calculate on the x_axis
    y_true = [f(i) for i in x_axis]
    ei = [EI(i, gp, min(y)) for i in x_axis]
    lcb = [LCB(i, gp, min(y), 1) for i in x_axis]

    # sklearn models expect data in the format [[x1], [x2], ...]
    m, s = gp.predict(x_axis.reshape(-1, 1), return_std=True)
    m = m.flatten()
    s = s.flatten()

    plt.figure(figsize=(7, 5))

    plt.subplot(211)
    plt.scatter(x, y)
    plt.plot(x_axis, y_true, label="true")
    plt.plot(x_axis, lcb, label="lcb")
    plt.plot(x_axis, m, label="model")
    plt.fill_between(x_axis, m - s, m + s, alpha=0.5)
    plt.legend()
    plt.ylabel("f(x)")
    plt.xlabel("x0")

    plt.subplot(212)
    plt.plot(x_axis, ei, label="ei")
    plt.legend()

    # append results to output directory
    save_dir = './outputs/'
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    plt.savefig(save_dir + "BO.png")
    plt.tight_layout()
    plt.show()

    results = {}
    results_file = "bo-results.pkl"
    if os.path.exists(results_file) and load_prior_results:
        with open(results_file, 'rb') as result_f:
            results = pickle.load(result_f)
        print("Loaded results from previous run.")
    else:
        # Actually run BO
        EI_res = []
        LCB_res = []
        rand_res = []
        grid_res = []

        for i in range(repetitions):
            iter_seed = seed + i
            bo_res_1 = run_bo(
                max_iter=num_evals,
                init=init,
                randomize=randomize,
                acquisition=EI,
                min_or_max="max",
                seed=iter_seed,
            )
            bo_res_2 = run_bo(
                max_iter=num_evals,
                init=init,
                randomize=randomize,
                acquisition=LCB,
                min_or_max="min",
                acq_params={"alpha": 1},
                seed=iter_seed,
            )
            r_res = run_random_search(max_iter=num_evals, seed=iter_seed)
            g_res = run_grid_search(max_iter=num_evals)

            # Get the minimums through the runs
            # [1, 4,5,6,  0, 8,9,10] -> [1,1,1,1,0,0,0,0]
            EI_res.append(np.minimum.accumulate(bo_res_1))
            LCB_res.append(np.minimum.accumulate(bo_res_2))
            rand_res.append(np.minimum.accumulate(r_res))
            grid_res.append(np.minimum.accumulate(g_res))

        results = (["EI", EI_res], ["LCB", LCB_res], ["rand", rand_res], ["grid", grid_res])

        # Save results to file
        with open(results_file, 'wb') as result_f:
            pickle.dump(results, result_f)
        print("Saved new results to file.")

    plt.figure(figsize=(7, 6))
    for name, res in results:
        r = np.vstack(res)
        m = np.mean(res, axis=0)
        s = np.std(r, axis=0)
        plt.step(np.arange(1, len(m)+1), m, label=name, where="post")
        plt.fill_between(np.arange(0, m.shape[0]), m + s, m - s, alpha=0.15, step="post")
        plt.grid(True, which="both", ls="-", alpha=0.3)

    plt.yscale("log")
    plt.ylabel("function value")
    plt.xlabel("function evaluations")
    plt.title("Comparison of Bayesian Optimization Methods")
    plt.legend()
    print("saving comparison")
    plt.savefig(save_dir + "BO_comp.png")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser("ex02")

    cmdline_parser.add_argument(
        "-n",
        "--num_func_evals",
        default=20,
        help="Number of function evaluations",
        type=int,
    )
    cmdline_parser.add_argument(
        "-p",
        "--percentage_init",
        default=0.25,
        help="Percentage of budget (num_func_evals) to spend on building initial model",
        type=float,
    )
    cmdline_parser.add_argument(
        "-r",
        "--random_initial_design",
        action="store_true",
        help="Use random initial points. If not set, initial points are sampled linearly on"
        " the function bounds.",
    )
    cmdline_parser.add_argument(
        "-v", "--verbose", default="INFO", choices=["INFO", "DEBUG"], help="verbosity"
    )
    cmdline_parser.add_argument(
        "--seed", default=0, help="Which seed to use", required=False, type=int
    )
    cmdline_parser.add_argument(
        "--repetitions",
        default=5,
        help="How often to repeat the experiment",
        required=False,
        type=int,
    )
    args, unknowns = cmdline_parser.parse_known_args()

    assert args.num_func_evals > 0
    assert 0.0 <= args.percentage_init <= 1.0
    main(
        num_evals=args.num_func_evals,
        init_size=args.percentage_init,
        repetitions=args.repetitions,
        randomize=args.random_initial_design,
        seed=args.seed,
    )
