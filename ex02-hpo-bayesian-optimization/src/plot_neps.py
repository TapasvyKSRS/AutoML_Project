import os
import yaml
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from main import f


def load_neps_results(base_dir="neps_results"):
    """Load NEPS results from specified base directory containing multiple task folders.

    Args:
        base_dir (str): The base directory containing task folders like 'seed=0', 'seed=42', etc.

    Returns:
        tuple: Mean and standard deviation of loss values across tasks.
    """
    all_losses = []
    task_dirs = [d for d in os.listdir(base_dir) if d.startswith('seed=') and d.split('=')[-1].isdigit()]
    task_dirs = sorted(task_dirs, key=lambda x: int(x.split('=')[-1]))

    for task_dir in task_dirs:
        task_path = Path(base_dir) / task_dir / "summary/"
        losses = []
        csv_files = sorted(os.listdir(task_path))
        for csv_file in csv_files:
            if 'full.csv' in csv_file:
                csv_path = task_path / csv_file
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df_sorted = df.sort_values(by=['time_sampled', 'time_end'])
                    losses.extend(df_sorted['objective_to_minimize'].tolist())
        all_losses.append(losses)
    # Calculate mean and standard deviation across tasks
    if all_losses:
        losses_array = np.array([np.minimum.accumulate(loss) for loss in all_losses])
        mean_losses = np.mean(losses_array, axis=0)
        # calculating the standard error of the mean
        sem_losses = np.std(losses_array, axis=0) / np.sqrt(len(mean_losses))
    else:
        mean_losses = sem_losses = np.array([])  # return empty arrays if no data

    return mean_losses, sem_losses, losses_array


def plot_neps_trajectory(
    neps_base_dir: str = "neps_results",
    bo_results_file: str = "bo-results.pkl",
    plot_name: str = "NePS-trajectory",
    solo_plot: bool = False,
    results: list = None
) -> None:
    """ Plot the trajectory of the NEPS optimizer and other optimizers.

    Args:
        neps_base_dir (str): Base directory containing NEPS results.
        bo_results_file (str): File containing results of other optimizers.
        plot_name (str): Name of the plot.
        solo_plot (bool): If True, plot only the NEPS trajectory.
        results (list): List of tuples containing optimizer name and results.
    """
    neps_mean = neps_sem = None
    x_plot_lim = 0
    if results is None:
        # Load NEPS results with uncertainties
        neps_mean, neps_sem, _ = load_neps_results(neps_base_dir)
        x_plot_lim = len(neps_mean)

        results = {}
        _bo_file_path = Path(__file__).parent.parent.absolute() / bo_results_file
        print(f"Loading results from: {_bo_file_path}")
        if os.path.exists(_bo_file_path):
            with open(_bo_file_path, 'rb') as f:
                results = pickle.load(f)
            print("Loaded results from file.")
        else:
            print("Results file not found.")
            raise ValueError("Results file not found. Run main.py successfully first.")

        results = results + (["NEPS", neps_mean], )

    if solo_plot:
        plt.figure()
    else:
        plt.figure(figsize=(12, 8))

    for name, res in results:
        if name == "NEPS":  # Special handling for NEPS with uncertainty
            plt.step(np.arange(1, len(res) + 1), res, label=f"{name}", where="post")
            plt.fill_between(
                np.arange(1, len(res) + 1),
                res - neps_sem,
                res + neps_sem,
                alpha=0.2,
                step="post"
            )
        elif not solo_plot:
            mean_res = np.mean(res, axis=0)
            std_dev_res = np.std(res, axis=0) / np.sqrt(len(res))  # calculating the standard error of the mean
            plt.step(np.arange(1, len(mean_res)+1), mean_res, label=f"{name}", where="post")
            plt.fill_between(
                np.arange(1, len(mean_res) + 1),
                mean_res + std_dev_res,
                mean_res - std_dev_res,
                alpha=0.2,
                step="post"
            )

    plt.xlim(0, x_plot_lim)
    plt.yscale("log")
    plt.ylabel("Minimum function value")
    plt.xlabel("Function evaluations")
    plt.title(plot_name)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    if not os.path.exists("./outputs"):
        os.mkdir("./outputs")
    plt.savefig("./outputs/"+plot_name+".png")
    plt.show()


if __name__ == "__main__":
    # Plot NEPS trajectory
    plot_neps_trajectory(
        neps_base_dir="./neps/",
        bo_results_file="bo-results.pkl",
        plot_name="NEPS-trajectory",
        solo_plot=False,
    )
