import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from src.nas_cifar10 import NASCifar10
from src.optimizers import Evolution
from src.optimizers import RandomSearch as RS

HERE = Path(__file__).parent.resolve()

LOGPATH = HERE / "outputs" / "logs"
RESULTS_PATH = LOGPATH / "results.obj"
PLOT_PATH = LOGPATH / "plot.png"

rcParams.update({"figure.autolayout": True})
plt.style.use("seaborn-v0_8-whitegrid")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_iters",
    default=100,
    type=int,
    nargs="?",
    help="number of iterations for optimization method",
)
parser.add_argument(
    "--benchmark_path",
    default="nasbench_only108.tfrecord",
    type=str,
    nargs="?",
    help="specifies the path to the tabular data",
)
parser.add_argument(
    "--pop_size", default=50, type=int, nargs="?", help="population size"
)
parser.add_argument(
    "--sample_size", default=10, type=int, nargs="?", help="sample_size"
)
parser.add_argument(
    "--n_repetitions",
    default=20,
    type=int,
    nargs="?",
    help="number of optimization independent runs",
)

args = parser.parse_args()


def main():
    # load the tabular benchmark
    b = NASCifar10(args.benchmark_path)

    # collect the results from each optimization run here
    rs_runs = []
    re_runs = []

    for i in range(args.n_repetitions):
        # run random search for n_iters function evaluations on benchmark b
        rs = RS(b)
        rs.optimize(args.n_iters)
        rs_runs.append(rs.incumbent_trajectory_error)
        print(f"RS run {i+1}/{args.n_repetitions} finished")

        # run regularized evolution for n_iters function evaluations on benchmark b
        re = Evolution(b)
        re.optimize(args.n_iters, args.pop_size, args.sample_size)
        re_runs.append(re.incumbent_trajectory_error)
        print(f"RE run {i+1}/{args.n_repetitions} finished")


    # plotting the incumbents error
    fig, ax = plt.subplots()
    x = range(args.n_iters)
    labels = ["RS", "RE"]
    all_runs = [rs_runs, re_runs]
    c = ["b", "r"]

    for i, setting in enumerate(all_runs):
        mean = np.mean(setting, axis=0)
        std = np.std(setting, axis=0)

        ax.plot(x, mean, c=c[i], label=labels[i])
        ax.fill_between(x, mean - std, mean + std, facecolor=c[i], alpha=0.3)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Valid. Error")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([5e-2, 7e-1])

    if not LOGPATH.exists():
        LOGPATH.mkdir(parents=True, exist_ok=True)

    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()

    results = {"re": re_runs, "rs": rs_runs}
    with RESULTS_PATH.open("wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
