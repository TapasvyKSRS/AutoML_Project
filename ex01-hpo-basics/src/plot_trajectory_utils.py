import os
import yaml
import numpy as np
from src.evolution import EA, ParentSelection, Mutation, Recombination
from src.target_function import ackley


def get_neps_trajectory(base_dir):

    """
    Extracts the x and y coordinates, and the loss values from the results of a 2D NEPS run.

    Also check the neps documentation for the 'post_run_summary' flag in neps.run
    for more convenient data extraction.
    """
    # Lists to hold our extracted data
    x_coordinates = []
    y_coordinates = []
    losses = []

    # Traverse through each configuration directory
    for config_dir in os.listdir(base_dir):
        config_path = os.path.join(base_dir, config_dir, "config.yaml")
        result_path = os.path.join(base_dir, config_dir, "report.yaml")

        # Ensure the paths exist before attempting to open
        if os.path.exists(config_path) and os.path.exists(result_path):
            # Read and extract data from config.yaml
            with open(config_path, 'r') as config_file:
                config_data = yaml.safe_load(config_file)
                x_coordinates.append(config_data["x_coordinate"])
                y_coordinates.append(config_data["y_coordinate"])

            # Read and extract data from result.yaml
            with open(result_path, 'r') as result_file:
                result_data = yaml.safe_load(result_file)
                losses.append(result_data["objective_to_minimize"])

    return x_coordinates, y_coordinates, losses


def get_ea_trajectory(opt_selection_type, opt_mutation_type, opt_recombination_type):
    """
    Run the EA algorithm with the specified hyperparameters and return the trajectory data.
    """
    ea = EA(
        target_func=ackley,
        population_size=20,
        problem_dim=2,
        selection_type=opt_selection_type,
        total_number_of_function_evaluations=500,
        problem_bounds=(-30, 30),
        mutation_type=opt_mutation_type,
        recombination_type=opt_recombination_type,
        sigma=1.0,
        children_per_step=5,
        fraction_mutation=0.5,
        recom_proba=0.5,
        logging=False
    )

    best_member = ea.optimize()

    # Collecting data for plots
    x_coordinates = [member.x_coordinate[0] for member in ea.all_members]
    y_coordinates = [member.x_coordinate[1] for member in ea.all_members]
    fitnesses = ea.fitness_values

    # EA Trajectory Plot
    iterations = list(range(len(ea.trajectory)))
    best_fitness_over_time = [member.fitness for member in ea.trajectory]

    results = {
        "x_coords": x_coordinates,
        "y_coords": y_coordinates,
        "fitnesses": fitnesses,
        "iterations": iterations,
        "best_fitness_over_time": best_fitness_over_time,
        "best_member": best_member
    }

    return results
