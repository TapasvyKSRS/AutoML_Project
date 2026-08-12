from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace


@dataclass
class Evaluation:
    y: float  # The value it got on a problem
    runtime: float  # The runtime to run this config


@dataclass
class HistItem:
    """Keeps track of cumulative metrics on the problem"""

    config: dict | Configuration  # The configuration tried
    y: float  # The value it got
    runtime: float  # The runtime of the config
    cumulative_runtime: float  # The cumulative runtime up to this trial
    regret_validation: float  # The regret of this config w.r.t. best before it
    regret_test: float | None = (
        None  # The regret of this config w.r.t. the best before it
    )


class Problem(ABC):
    def __init__(self, seed: int | None):
        self.seed = seed
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        self.evaluations: list[Tuple[dict, Evaluation]] = []

    @abstractmethod
    def objective_function(
        self,
        config: dict | Configuration,
        budget: int,
    ) -> Evaluation:
        """Evaluate the config, giving back its value on the problem and it's runtime

        Parameters
        ----------
        config: dict | Configuration
            The configuration to try out

        budget: int
            The allocated budget in seconds for this configuration

        Returns
        -------
        Evaluations
        """
        ...

    @classmethod
    @abstractmethod
    def get_configuration_space(cls) -> ConfigurationSpace:
        """Get the configuration space for this problem

        Returns
        -------
        ConfigurationSpace
            The config space for this problem
        """
        ...

    @abstractmethod
    def history(self) -> list[HistItem]:
        """Get the history of all evaluations"""
        ...

    def reset(self) -> None:
        """Reset the problem instance and all the evaluations it remembers"""
        self.evaluations = []
        self.rng = np.random.RandomState(self.seed)
