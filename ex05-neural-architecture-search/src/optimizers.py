from __future__ import annotations

from collections import deque
from copy import deepcopy

import math
import numpy as np
import random
from ConfigSpace import Configuration, OrdinalHyperparameter

from src.nas_cifar10 import NASCifar10
from src.predictor import EnsemblePredictor


class NASOptimizer:
    """Base class for NASBench-101 optimizers"""

    def __init__(self, benchmark: NASCifar10):
        # get the configuration space
        self.benchmark = benchmark
        self.cs = benchmark.get_configuration_space()
        self.curr_wallclock = 0.0
        self.curr_incumbent = None
        self.curr_incumbent_error = np.inf

        # incumbent_trajectory keeps track of the best
        # configuration (architecture) at each point in time.
        self.incumbent_trajectory: list[Configuration] = []

        # incumbent_trajectory_error keeps track of the
        # corresponding validation errors of incumbent_trajectory
        self.incumbent_trajectory_error: list[float] = []

    def optimize(self, n_iters: int = 100) -> None:
        """Optimize for some amount of iterations

        The results will be in:
        * `self.incumbent_trajectory`
        * `self.incumbent_trajectory_error`

        Parameters
        ----------
        n_iters : int
            The amount of iterations to optimize for
        """
        raise NotImplementedError()

    def sample_random_config(self) -> Configuration:
        """Return a randomly sampled configuration.

        Returns
        -------
        Configuration
            The randomly sampled configuration
        """
        # TODO: return one randomly sampled configuration from self.cs
        return self.cs.sample_configuration()

    def train_and_eval(self, config: Configuration) -> tuple[float, float]:
        """Queries the validation error of config in self.benchmark.

        Since every architecture has already been trained and evaluated, we just
        do table look-ups without the need to train the neural net.

        Parameters
        ----------
        config : Configuration
            The configuration to query

        Returns
        -------
        (validation_error: float, cost: float)
            Validation error and cost of the config
        """
        y = 0.0
        cost = 0.0
        # TODO: query validation error from the tabular benchmark
        y, cost = self.benchmark.objective_function(config)
        self.curr_wallclock += cost
        # TODO: check if config is better than current incumbent
        if y < self.curr_incumbent_error:
            self.curr_incumbent = config
            self.curr_incumbent_error = y
        # TODO: updated the incumbent trajectory
        self.incumbent_trajectory.append(self.curr_incumbent)
        self.incumbent_trajectory_error.append(self.curr_incumbent_error)

        # return the validation error and cost of the queried config
        return y, cost


class RandomSearch(NASOptimizer):
    """Algorithm for random search."""

    def optimize(self, n_iters: int = 100) -> None:
        """Run random search

        Parameters
        ----------
        n_iters : int
            The amount of function evaluations
        """
        for i in range(n_iters):
            config = self.sample_random_config()
            self.train_and_eval(config)


class Evolution(NASOptimizer):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".
    """

    def __init__(self, benchmark: NASCifar10):
        super().__init__(benchmark)
        self.population: deque[Configuration] = deque()

        # Dictionary from a configuration to its validation error
        self.history: dict[Configuration, float] = {}

    def optimize(
        self,
        n_iters: int = 100,
        population_size: int = 100,
        sample_size: int = 20,
        regularized: bool = True,
    ) -> None:
        """

        Parameters
        ----------
        n_iters : int = 100
            The number of iterations the algorithm should run for.

        population_size : int = 100
            The number of individuals to keep in the population.

        sample_size : int = 20
            The number of individuals that should participate in each tournament.

        regularized : bool = True
            Whethere it should do so regularized or not
        """
        # Initialize the population with random models.
        while len(self.population) < population_size:
            config = self.sample_random_config()
            y, _ = self.train_and_eval(config)

            self.population.append(config)
            self.history[config] = y

        # Carry out evolution in cycles (n_iters). Each cycle produces a model
        # and removes another.
        while len(self.history) < n_iters:
            # Sample randomly chosen models from the current population.
            # Config -> validation error
            sample: dict[Configuration, float] = {}

            while len(sample) < sample_size:
                # TODO: randomly choose one architecture from population and add
                # it to sample
                candidate = random.choice(list(self.population))
                sample[candidate] = self.history[candidate]

            # TODO: get the best model in the current sample
            parent = min(sample, key=lambda c: sample[c])
            # TODO: Create the child model and evaluate it.
            # NOTE: Child model can be the same as one of the previously evaluated
            # models. Do not evalutate it if the same model has already been evaluated.
            child = self.mutate_arch(parent)
            if child not in self.history:
                y, _ = self.train_and_eval(child)
                self.history[child] = y
            else:
                y = self.history[child]

            # TODO: add the child to the population and history
            self.population.append(child)
            self.history[child] = y
            # TODO: based on the regularized argument value remove the oldest or the
            # worst architecture from the population
            if regularized:
                self.population.popleft()
            else:
                worst = max(self.population, key=lambda c: self.history[c])
                self.population.remove(worst)

    def mutate_arch(self, parent_arch: Configuration) -> Configuration:
        """Apply mutation to a parent architecture and return the mutated one.

        Parameters
        ----------
        parent_arch : Configuration
            The parent configuration

        Returns
        -------
        Configuration
            The mutated child
        """
        # pick random parameter
        dim = np.random.randint(len(self.cs.get_hyperparameters()))

        hyper = self.cs.get_hyperparameters()[dim]
        if isinstance(hyper, OrdinalHyperparameter):
            choices = list(hyper.sequence)
        else:
            choices = list(hyper.choices)

        # drop current values from potential choices
        choices.remove(parent_arch[hyper.name])

        # flip parameter
        idx = np.random.randint(len(choices))
        child_arch = deepcopy(parent_arch)
        child_arch[hyper.name] = choices[idx]

        return child_arch
