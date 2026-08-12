import neps
import numpy as np
from pathlib import Path
import random
import argparse
from main import f as objective_function
import torch
import warnings
torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", message=".*added jitter of.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="botorch")


def set_seed(seed: int):
    """
    Set the seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


#  complete the following function to define the pipeline for the NePS framework.
def run_neps_pipeline(seed=42, max_evals=50, root_directory="./neps"):
    """
    Run the NePS pipeline.
    """

    # TODO: Use the parameter 'x' with a search range of [-15, 10] in the pipeline_space
    pipeline_space = {'x': neps.Float(lower=-15, upper=10)}
    # TODO: Set the seed for reproducibility
    set_seed(seed)
    # Complete the NePS pipeline
    neps.run(
        evaluate_pipeline=objective_function,  # TODO set target function pipeline
        root_directory=root_directory,
        pipeline_space=pipeline_space,  # TODO set the pipeline space
        max_evaluations_total=max_evals,
        optimizer='bayesian_optimization',
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NePS with multiple seeds")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--max_evals", type=int, default=20, help="Maximum evaluations per seed")
    parser.add_argument("--root_directory", type=str, default="./neps", help="Root directory for NEPS results")

    args = parser.parse_args()
    seed = args.seed
    max_evals = args.max_evals
    root_directory = Path(args.root_directory) / f"seed={seed}"

    # TODO: Run the NEPS pipeline with Levy1D objective function
    run_neps_pipeline(seed=seed, max_evals=max_evals, root_directory=root_directory)

    # TODO: Using the curated pipeline, run NePS over multiple seeds.
    for seed in [42, 43, 44, 45, 46]:
        run_neps_pipeline(seed=seed, max_evals=max_evals, root_directory=root_directory / f"seed={seed}")
