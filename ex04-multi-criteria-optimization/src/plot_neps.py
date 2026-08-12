import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.multicriterion_opt import eval_dt_config
from src.pareto import construct_polygon, pareto, computeHV2D


def load_neps_results(neps_base_dir: str = "neps") -> tuple:
    """Load NEPS results from a nested directory structure and organize results in a dictionary.

    Args:
        neps_base_dir (str): The base directory containing 'optimizer=.../method=.../seed=...' folders.

    Returns:
        tuple: A tuple containing two dictionaries:
            - results: A dictionary with keys as 'optimizer=.../method=...' and values as tuples of
                (mean_losses, sem_losses, all_losses).
            - configs: A dictionary with keys as 'optimizer=.../method=...' and values as DataFrames of configurations.
    """
    results = {}
    configs = {}
    base_path = Path(neps_base_dir)
    if not base_path.exists():
        raise ValueError(f"Base directory {neps_base_dir} does not exist.")

    # Traverse the directory structure
    for optimizer_dir in base_path.glob("optimizer=*"):
        for method_dir in optimizer_dir.glob("method=*"):
            all_losses = []
            task_key = f"{optimizer_dir.name}/{method_dir.name}"
            print(f"Processing {task_key}...")
            for seed_dir in method_dir.glob("seed=*"):
                task_path = seed_dir / "summary/"
                if not task_path.exists():
                    continue

                losses = []
                csv_files = sorted(task_path.glob("*full.csv"))
                for csv_path in csv_files:
                    if csv_path.exists():
                        df = pd.read_csv(csv_path)
                        df_sorted = df.sort_values(by=['time_sampled', 'time_end'])
                        losses.extend(df_sorted['objective_to_minimize'].tolist())
                        configs[task_key] = pd.concat([
                            configs.get(task_key, pd.DataFrame()),
                            df_sorted[['config.max_leaf_nodes', 'config.max_depth']]
                        ], ignore_index=True)
                all_losses.append(losses)

            # Calculate mean and standard error of the mean (SEM) for the losses
            if all_losses:
                losses_array = np.array([np.minimum.accumulate(loss) for loss in all_losses])
                mean_losses = np.mean(losses_array, axis=0)
                sem_losses = np.std(losses_array, axis=0) / np.sqrt(len(mean_losses))  # Standard Error of the Mean
                results[task_key] = (mean_losses, sem_losses, losses_array)
            else:
                results[task_key] = (np.array([]), np.array([]), np.array([]))

    return results, configs


def plot_neps_trajectories(
    neps_base_dir: str = "neps",
    plot_name: str = "NePS-scalarization",
    results: list = None
) -> None:
    """ Plot the NEPS trajectories for different optimizers and methods.

    Args:
        neps_base_dir (str): Base directory containing NEPS results.
        plot_name (str): Name of the plot.
        results (list): List of tuples containing optimizer name and results.
    """
    x_plot_lim = 0
    if results is None:
        # Load NEPS results with uncertainties
        results, _ = load_neps_results(neps_base_dir)

    plt.figure(figsize=(12, 8))

    for key, (mean_losses, sem_losses, _) in results.items():
        x_plot_lim = max(x_plot_lim, mean_losses.shape[0])
        plt.step(np.arange(1, len(mean_losses) + 1), mean_losses, label=f"{key}", where="post")
        plt.fill_between(
            np.arange(1, len(mean_losses) + 1),
            mean_losses - sem_losses,
            mean_losses + sem_losses,
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

    os.makedirs("./outputs", exist_ok=True)
    plt.savefig(f"./outputs/{plot_name}.png")
    plt.show()


def plot_pareto_fronts_with_hv(
    ref_point: tuple = (1.0, 1.0),
    plot_name: str = "pareto_fronts_hv",
) -> None:
    """Plot Pareto fronts with hypervolumes.

    Args:
        ref_point (tuple): Reference point for hypervolume calculation.
        plot_name (str): Name of the plot.
    """
    _, configs = load_neps_results()
    hypervolumes = {}

    all_polygons = []
    all_pareto_fronts = []

    task_keys = list(configs.keys())
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(task_keys))]
    task_to_color = {task: colors[i] for i, task in enumerate(task_keys)}

    for task in task_keys:
        config = configs[task]
        # TODO: Evalute the configurations
        evals = []
        for _, row in config.iterrows():
            max_leaf_nodes = int(row['config.max_leaf_nodes'])
            max_depth = int(row['config.max_depth'])
            error, runtime = eval_dt_config(max_leaf_nodes, max_depth, k=1)
            evals.append((error, runtime))
        evals = np.array(evals)
        # raise NotImplementedError("Evaluation of configurations is not implemented.")
        # TODO: Compute the pareto front and the hypervolume for that pareto front
        pareto_front = pareto(evals)
        pareto_front = evals[pareto_front]
        pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]
        polygon = construct_polygon(pareto_front, ref_point)
        hv = computeHV2D(pareto_front, ref_point)
        # raise NotImplementedError("Polygon construction is not implemented.")
        all_polygons.append((polygon, task, hv))
        all_pareto_fronts.append((pareto_front, task))

        color = task_to_color[task]

        plt.figure(figsize=(10, 8))
        plt.fill(polygon[:, 0], polygon[:, 1], alpha=0.1, color=color, label=f"{task} (HV={hv:.4f})")
        plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'o', color=color, label=f"{task} pareto points")
        plt.scatter(*ref_point, color='red', marker='x', s=100, label="Reference Point")
        plt.xlabel("Error Rate")
        plt.ylabel("Runtime")
        plt.title(f"Pareto Front: {task}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(12, 9))
    for (polygon, task, hv), (pareto_front, _) in zip(all_polygons, all_pareto_fronts):
        color = task_to_color[task]
        plt.fill(polygon[:, 0], polygon[:, 1], alpha=0.1, color=color, label=f"{task} (HV={hv:.4f})")
        plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'o', color=color)

    plt.scatter(*ref_point, color='red', marker='x', s=100, label="Reference Point")
    plt.xlabel("Error Rate")
    plt.ylabel("Runtime")
    plt.title("All Pareto Fronts with Shaded Hypervolumes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    Path("outputs").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"outputs/{plot_name}.png")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hypervolume",
        action="store_true",
        help="Plot hypervolume of Pareto fronts.",
    )
    args = parser.parse_args()

    if args.hypervolume:
        plot_pareto_fronts_with_hv()
    else:
        plot_neps_trajectories(
            neps_base_dir="./neps/",
            plot_name="NEPS-scalarization",
        )
