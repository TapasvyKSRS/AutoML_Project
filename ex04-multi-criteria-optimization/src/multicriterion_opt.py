from pathlib import Path
import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

from src.pareto import pareto, nDS

np.random.seed(1000)
HERE = Path(__file__).parent.resolve()

# load the spam dataset
spam = fetch_openml("spambase")
X, y = spam.data, spam.target


def eval_dt_config(
    max_leaf_nodes: int = 10,
    max_depth: int = 10,
    k: int = 1
) -> tuple[float, float]:
    """Runs k-fold cross validation on the spam classification task using a decision tree
    classifier, and returns the average misclassification rate and training time.

    Parameters
    ----------
    max_leaf_nodes : int
        Maximum number of leaf nodes in the tree.

    max_depth : int
        The maximum depth of the tree.

    k : int, optional
        Number of folds, by default 1

    Returns
    -------
    float, float
        Mean misclassification rate and mean training time.
    """
    misclassif_rates = []
    train_times = []

    for _ in range(k):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = DecisionTreeClassifier(max_depth=max_depth,
                                       max_leaf_nodes=max_leaf_nodes)
        start = time.time()
        model.fit(X_train, y_train)
        train_times.append(time.time() - start)
        misclassif_rates.append(1 - model.score(X_test, y_test))

    return np.mean(misclassif_rates), np.mean(train_times)


def multicriterion_random_search(
    pop_size: int = 200,
    range_max_leaf_nodes: tuple[int, int] = (2, 30),
    range_max_depth: tuple[int, int] = (2, 30)
) -> list:
    """Runs a random search on the decision tree classifier, aiming to minimize
    the misclassification rate and the training time simultaneously.

    Parameters
    ----------
    pop_size : int, optional
        Population size, by default 200

    range_max_leaf_nodes : tuple[int, int], optional
        Range of the maximum number of leaf nodes, by default (2, 30)

    range_max_depth : tuple[int, int], optional
        Range of the maximum depth, by default (2, 30)

    Returns
    -------
    list
        Pareto front of the population.
    """

    # TODO : create a population of random configurations and evaluate them
    # --------------
    population = np.zeros((pop_size, 2), dtype=int)
    for i in range(pop_size):
        population[i][0] = np.random.randint(*range_max_leaf_nodes)
        population[i][1] = np.random.randint(*range_max_depth)
    # --------------

    # TODO : compute the pareto front of the population
    # Hint : use the pareto function from pareto.py
    # --------------
    fit = np.array([eval_dt_config(*ind) for ind in population])
    pareto_front = fit[pareto(fit)]
    # --------------

    # plot the pareto front
    fig, ax = plt.subplots()
    ax.scatter([f[0] for f in fit], [f[1] for f in fit], color="gray", label="dominated solutions")
    ax.scatter([f[0] for f in pareto_front], [f[1] for f in pareto_front], color="blue", label="pareto front")

    plt.xlabel("Misclassification rate")
    plt.ylabel("Training time")

    plt.legend(fontsize='small')
    plt.title("multiobjective random search on the decision tree classifier")
    Path("outputs").mkdir(parents=True, exist_ok=True)
    plt.savefig("outputs/pareto_front_decision_tree_random_search.png")
    plt.show()

    return pareto_front


def multicriterion_evolutionary_search(
    pop_size: int = 40,
    range_max_leaf_nodes: tuple[int, int] = (2, 30),
    range_max_depth: tuple[int, int] = (2, 30),
    nb_generations: int = 5,
) -> None:
    """Runs an evolutionary search on the decision tree classifier, aiming to minimize
    the misclassification rate and the training time simultaneously.

    Parameters
    ----------
    pop_size : int, optional
        Population size, by default 40

    range_max_leaf_nodes : tuple[int, int], optional
        Range of the maximum number of leaf nodes, by default (2, 30)

    range_max_depth : tuple[int, int], optional
        Range of the maximum depth, by default (2, 30)

    nb_generations : int, optional
        Number of generations of the evolutionary search, by default 5
    Returns
    -------
    list
        List of pareto fronts of the successive populations.
    """

    # TODO : create a population of random configurations and evaluate them
    # --------------
    population = np.zeros((pop_size, 2), dtype=int)
    for i in range(pop_size):
        population[i][0] = np.random.randint(*range_max_leaf_nodes)
        population[i][1] = np.random.randint(*range_max_depth)
    # --------------

    pareto_fronts = []

    # run the evolutionary search

    for i in range(nb_generations):
        print("generation = {}".format(i))

        # create children

        children = np.zeros(population.shape)

        for j in range(pop_size):
            # crossover
            k_ind = np.random.randint(0, pop_size)
            l_ind = np.random.randint(0, pop_size)
            while k_ind == l_ind:
                l_ind = np.random.randint(0, pop_size)
            children[j] = (population[k_ind] + population[l_ind])/2

            # random chance of mutation for each parameter
            for k, (lower, upper) in enumerate([range_max_leaf_nodes, range_max_depth]):
                if np.random.rand() < 0.1:
                    children[j][k] = np.random.randint(lower, upper + 1)

        # add children to the population

        population = np.concatenate((population, children), axis=0).astype(int)

        # evaluate population

        fit = np.array([eval_dt_config(*ind) for ind in population])

        # compute the pareto front of the fitness values
        pareto_front = fit[pareto(fit)]
        pareto_front = np.sort(
            pareto_front.view([("", pareto_front.dtype)] * pareto_front.shape[1]),
            order=["f1"],  # order by first element then second and so on
            axis=0,
        ).view(float)

        pareto_fronts.append(pareto_front)

        # TODO: select the best pop_size individuals, using non-dominated sorting
        '''
        **Simplified** NSGA-II survivor selection:
        # We preserve the core NSGA-II ranking behavior: candidates are ordered by
        # non-dominated sorting, and complete Pareto fronts are added in rank order.
        # If the next front would exceed pop_size, we keep only as many individuals
        # from that front as needed to fill the remaining slots.
        # In contrast to full NSGA-II, we do not apply crowding-distance
        # tie-breaking within the split front, so diversity along that front may be lower.
        '''
        # --------------
        new_population = []
        remaining_indices = np.arange(len(fit))
        remaining_fit = fit.copy()
        while len(remaining_fit) > 0 and len(new_population) < pop_size:
            mask = pareto(remaining_fit)
            front_indices = remaining_indices[mask]

            slots = pop_size - len(new_population)
            if len(front_indices) <= slots:
                new_population.extend(population[front_indices])
            else:
                new_population.extend(population[front_indices[:slots]])
                break
            remaining_fit = remaining_fit[~mask]
            remaining_indices = remaining_indices[~mask]
        population = np.array(new_population, dtype=int)

    # plot the successive pareto fronts

    for i, pf in enumerate(pareto_fronts):

        plt.scatter(pf[:, 0], pf[:, 1],
                    label="pareto front gen. {}".format(i),
                    color=(0, (i+1)/len(pareto_fronts), 0))

    plt.xlabel("Misclassification rate")
    plt.ylabel("Training time")

    plt.legend(fontsize='small')
    plt.title("successive pareto fronts of the multiobjective evolutionary search")
    Path("outputs").mkdir(parents=True, exist_ok=True)
    plt.savefig("outputs/pareto_front_decision_tree_evolutionary_search.png")
    plt.show()

    # --------------
    return pareto_fronts


if __name__ == "__main__":
    # run the random search and the evolutionary search, using the same number of evaluations

    nb_evaluations = 200
    nb_generations = 5

    pop_size_random_search = nb_evaluations
    pop_size_evolutionary_search = nb_evaluations // nb_generations

    pareto_front_random = multicriterion_random_search(pop_size=pop_size_random_search)
    pareto_fronts_evolutionary = multicriterion_evolutionary_search(pop_size=pop_size_evolutionary_search,
                                                                    nb_generations=nb_generations)

    # plot both pareto fronts on the same figure
    fig, ax = plt.subplots()
    ax.scatter([f[0] for f in pareto_front_random],
               [f[1] for f in pareto_front_random],
               c="blue",
               label="random search pareto front")

    ax.scatter([f[0] for f in pareto_fronts_evolutionary[-1]],
               [f[1] for f in pareto_fronts_evolutionary[-1]],
               c="green",
               label="evolutionary search pareto front")
    plt.xlabel("Misclassification rate")
    plt.ylabel("Training time")

    plt.legend(fontsize='small')
    plt.title("pareto fronts of the random search and the evolutionary search on the decision tree classifier")
    Path("outputs").mkdir(parents=True, exist_ok=True)
    plt.savefig("outputs/pareto_front_random_vs_evolutionary.png")
    plt.show()
