import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from src.plot_trajectory_utils import get_neps_trajectory, get_ea_trajectory
from src.evolution import ParentSelection, Mutation, Recombination


def plot_neps_results(root_dir: str, save_path: str = "./neps_plot.png"):
    """
    Plots NEPS search trajectory and loss over iterations.

    Args:
        root_dir (str): Path to the NEPS results directory.
        save (bool): Whether to save the plot as a file.
        save_path (str): Path to save the plot if save is True.
    """
    x_coordinates, y_coordinates, losses = get_neps_trajectory(root_dir)

    neps_min_losses = np.minimum.accumulate(losses)
    neps_iterations = list(range(len(neps_min_losses)))

    plt.figure(figsize=(14, 6))

    # Search Space Exploration Plot
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        x_coordinates, y_coordinates, c=losses,
        cmap='viridis', alpha=np.linspace(0.25, 1, len(losses))
    )
    plt.colorbar(scatter, label='Loss')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Search Space Exploration')

    # Loss over Iterations Plot
    plt.subplot(1, 2, 2)
    plt.plot(neps_iterations, neps_min_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Value by Iteration')
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved at {save_path}")
    else:
        plt.show()


def plot_ackley_comparison(save_path: str = "./neps_comparison_plot.png"):
    """
    Assuming you have already finished coding the EA functions,
    we can also visualize its new config selection and evaluate its performance.
    """

    # TODO: Load the trajectory data from your EA run.
    best_mutation = Mutation.GAUSSIAN
    best_selection = ParentSelection.TOURNAMENT
    best_recombination = Recombination.INTERMEDIATE
    # - Extract the following from your EA implementation:
    #     - x_coordinates: X1 values of all evaluated solutions
    #     - y_coordinates: X2 values of all evaluated solutions
    #     - fitnesses: Fitness/Loss values corresponding to each solution
    #     - iterations: Evaluation step or generation index
    #     - best_fitness_over_time: Best fitness achieved so far at each iteration
    #     - best_member: The best solution found after the EA run
    ea_trajectory = get_ea_trajectory(
        opt_selection_type=best_selection,
        opt_mutation_type=best_mutation,
        opt_recombination_type=best_recombination,
    )
    x_coordinates, y_coordinates, fitnesses = (
        ea_trajectory["x_coords"],
        ea_trajectory["y_coords"],
        ea_trajectory["fitnesses"])
    iterations, best_fitness_over_time, best_member = (
        ea_trajectory["iterations"],
        ea_trajectory["best_fitness_over_time"],
        ea_trajectory["best_member"])

    # Plotting
    plt.figure(figsize=(14, 6))

    # Search Space Exploration
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        x_coordinates, y_coordinates, c=fitnesses,
        cmap='viridis', alpha=np.linspace(0.3, 1, len(fitnesses))
    )
    plt.colorbar(scatter, label='Fitness')
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.xlabel('X1 Coordinate')
    plt.ylabel('X2 Coordinate')
    plt.title('Search Space Exploration')

    # Fitness Over Time
    plt.subplot(1, 2, 2)
    plt.plot(iterations, best_fitness_over_time, marker='o', linestyle='-', color='b')
    plt.xlabel('Children idx')
    plt.ylabel('Best Fitness')
    plt.title('Loss Value by Iteration')
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved at {save_path}")
    else:
        plt.show()

    num_children = len(fitnesses)
    num_iterations = len(best_fitness_over_time)
    min_loss = best_member.fitness

    print(f"Plotted {num_children} children evaluations over {num_iterations} iterations.")
    print(f"Minimum Loss: {min_loss}")


if __name__ == "__main__":
    # TODO: Base directory for NEPS results where the configs are stored
    root_dir = "./neps/results_ackley/configs"

    # TODO: Plot NEPS results and save the plot to outputs directory
    plot_neps_results(root_dir=root_dir, save_path="./outputs/neps-on-ackley.png")
    # TODO: Plot EA results and save the plot to outputs directory
    plot_ackley_comparison(save_path="./outputs/ackley-comparison.png")
