import argparse
from pathlib import Path
import os
import time
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt

import neps

from src.fcnet_benchmark import FCNetBenchmark


def run_neps(
    benchmark: FCNetBenchmark,
    optimizer: str = "hyperband"
):
    # TODO: Create Pipeline Space
    # 1. Take reference from the `get_configuration_space` method in `FCNetBenchmark` and
    # 2. Convert it to a NEPS suitable pipeline space as in the previous exercises.
    # 3. Also add the 'epoch' as a parameter to the pipeline space.
    # 4. The 'epoch' parameter should be in [10, 100] and must be treated as a Fidelity parameter.
    # We recommend the NePS docs as your main reference: https://automl.github.io/neps/0.13.0/reference/pipeline_space/
    pipeline_space = {
        "n_units_1": neps.Categorical(choices=[16, 32, 64, 128, 256, 512]),
        "n_units_2": neps.Categorical(choices=[16, 32, 64, 128, 256, 512]),
        "dropout_1": neps.Categorical(choices=[0.0, 0.3, 0.6]),
        "dropout_2": neps.Categorical(choices=[0.0, 0.3, 0.6]),
        "activation_fn_1": neps.Categorical(choices=["tanh", "relu"]),
        "activation_fn_2": neps.Categorical(choices=["tanh", "relu"]),
        "init_lr": neps.Categorical(choices=[5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]),
        "lr_schedule": neps.Categorical(choices=["cosine", "const"]),
        "batch_size": neps.Categorical(choices=[8, 16, 32, 64]),
        "epoch": neps.Integer(lower=10, upper=100, is_fidelity=True),
    }

    def run_pipeline(**config):
        # We expect you define a parameter named 'epoch' in the pipeline space
        # which will be used as the budget for evaluation.
        budget = config.pop('epoch')
        evaluation = benchmark.objective_function(config, budget)

        # time.sleep(evaluation.runtime / 10)  # NOTE: uncomment only for parallel runs

        return {
            "objective_to_minimize": evaluation.y,
            "loss": -evaluation.y,
            "cost": evaluation.runtime,
            "info_dict": {
                "pid": os.getpid(),
            }
        }

    # TODO: Set the NePS pipeline to store optimizer specific results
    # You can modify the root directory to store the results in a specific folder
    neps.run(
        evaluate_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory=f"./neps/{optimizer}",
        max_evaluations_total=50,
        optimizer=optimizer,
        overwrite_working_directory=False,
        post_run_summary=True,
    )


def plot_neps(
    plt,
    root_directory="./neps/",
    task_name: str = "hyperband",
    log_x: bool = False,
    log_y: bool = True,
):
    # load the results from the task
    df = pd.read_csv(f"{root_directory}/{task_name}/summary/full.csv", index_col=0)

    # sort by time sample finished evaluation, this order is accurate even for parallel evaluations
    df = df.sort_values(by=["time_sampled", "time_end"])

    # plot the loss over time
    plt.plot(
        df["cost"].cumsum().values,
        np.minimum.accumulate(df["objective_to_minimize"]).values,
        label=task_name
    )
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.xlabel("(Simulation) Runtime Cost [in s]")
    plt.ylabel("Loss")
    plt.legend()

    return plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed for the run(s)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="all",
        choices=["all", "successive_halving", "hyperband"],
        help="The optimizer to use for the search",
    )
    parser.add_argument(
        "--root_directory",
        type=str,
        default="./neps/",
        help="The directory for NePS output files",
    )
    parser.add_argument(
        "--plot_directory",
        type=str,
        default="./outputs/",
        help="The directory to save plots",
    )
    parser.add_argument(
        "--only_plot",
        action="store_true",
        help="To skip NePS run and only trigger plotting",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="To show logs in stdout at INFO level",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Run NePS
    if not args.only_plot:
        if args.run == "all" or args.run == "successive_halving":
            np.random.seed(args.seed)
            run_neps(FCNetBenchmark(name="protein_structures"), optimizer="successive_halving")

        if args.run == "all" or args.run == "hyperband":
            np.random.seed(args.seed)
            run_neps(FCNetBenchmark(name="protein_structures"), optimizer="hyperband")

    # Plotting
    plt.clf()
    if args.run == "all" or args.run == "successive_halving":
        plot_neps(
            plt,
            root_directory=args.root_directory,
            task_name=f"successive_halving",
            log_y=False,
        )
    if args.run == "all" or args.run == "hyperband":
        plot_neps(
            plt,
            root_directory=args.root_directory,
            task_name=f"hyperband",
            log_y=False,
        )
    save_suffix = "" if args.run == "all" else f"_{args.run}"
    plt.savefig(Path(args.plot_directory) / f"neps_results{save_suffix}.pdf", dpi=300)
