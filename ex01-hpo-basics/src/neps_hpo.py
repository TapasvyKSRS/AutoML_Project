import neps
import numpy as np
import logging
from src.target_function import ackley


def run_ackley_pipeline(**config):

    # read the parameters from the provided configuration
    ack_parameter = np.array([config["x_coordinate"], config["y_coordinate"]])

    loss = ackley(ack_parameter)
    return loss


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    # TODO: 1. Define pipeline space in correct format
    # 2. Fill out the missing parameters (...) of neps.run()
    # use the neps documentation to clarify the parameters
    # 3. Then run neps_hpo.py and check the plots in plot_neps.ipynb
    pipeline_space = dict(
        x_coordinate=neps.Float(lower=-30, upper=30),
        y_coordinate=neps.Float(lower=-30, upper=30)
    )
    # TODO: when working with neps, you only have to provide it with the target function
    #       and the configuration space
    #       Additionally, you can also change the root_directory to a new location, set the evaluations to spend, etc
    neps.run(
        evaluate_pipeline=run_ackley_pipeline,
        pipeline_space=pipeline_space,
        root_directory="./neps/results_ackley",
        evaluations_to_spend=100,
    )
    # TODO: Define the pipeline_space.
    # Use neps.Float(lower=..., upper=...) for both 'x_coordinate' and 'y_coordinate'.
    # The search range for Ackley is typically [-30, 30] for both the coordinates.
    pipeline_space = dict(
        x_coordinate=neps.Float(lower=-30, upper=30),
        y_coordinate=neps.Float(lower=-30, upper=30),
    )

    # Convert YAML to neps-compatible format

    neps.run(
        evaluate_pipeline=run_ackley_pipeline,  # callable function which has chosen HPs config as input
        # and returns the loss of the config
        pipeline_space=pipeline_space,  # config space as dictionary of HPs to be optimized and their search space
        root_directory="./neps/results_ackley",  # choose a directory where the results will be stored
        evaluations_to_spend=100,
    )
