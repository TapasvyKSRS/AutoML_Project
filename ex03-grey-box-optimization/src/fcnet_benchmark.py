from __future__ import annotations

import json
from itertools import accumulate
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace
)

from src.problem import Evaluation, HistItem, Problem

HERE = Path(__file__).parent.resolve()
DATADIR = HERE.parent / "data"

TOTAL_BUDGET = 100

datasets = {
    "protein_structures": DATADIR / "fcnet_protein_structure_data.hdf5",
    "naval_propulsion": DATADIR / "fcnet_naval_propulsion_data.hdf5",
    "parkinsons": DATADIR / "fcnet_parkinsons_telemonitoring_data.hdf5",
    "slice_localization": DATADIR / "fcnet_slice_localization_data.hdf5",
}


class FCNetBenchmark(Problem):
    """A wrapper around previously recorded configuration runs on a set of different
    problems.

    By using the `config` as a key, we can query the data for the different performance
    metrics of how it performed.
    """

    # Some information used in creating the dataset
    total_budget = TOTAL_BUDGET

    def __init__(
        self,
        name: str = "protein_structures",
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        name : "protein_structures" | "naval_propulsion" | "parkinsons" | "slice_localization"
            Name of the dataset to load

        seed : int = None
            The seed to use
        """
        super().__init__(seed)

        if name not in datasets:
            raise ValueError(f"`name` ({name}) must be one of {set(datasets.keys())}")

        dataset_path = str(datasets[name])
        self.data = h5py.File(dataset_path, "r")

    def best_config(self) -> Tuple[dict, float, float]:
        """The configuration in the dataset that achieved the lowest test performance.

        Returns
        -------
        (best_config, test_error, valid_mse)
        """
        test_error = lambda config: np.mean(self.data[config]["final_test_error"])  # noqa
        mse = lambda config: np.mean(self.data[config]["valid_mse"][:, -1])  # noqa

        # Find the best configuration, the one which achieved the lowest test error
        best_config = min(self.data, key=test_error)

        # Get it's test error and it's mse then
        best_error = test_error(best_config)
        best_mse = mse(best_config)

        return (best_config, best_error, best_mse)

    def objective_function(
        self,
        config: dict | Configuration,
        budget: int = TOTAL_BUDGET,
        index: int | None = None,
    ) -> Evaluation:
        """Evaluate a configuration.

        Does this by querying the dataset to get its performance.

        Parameters
        ----------
        config : dict | Configuration
            The configuration to query

        budget : int = TOTAL_BUDGET
            At what budget to get the evaluation from

        index : int | None = None
            The index of the run to choose. There are 4 runs in total so you can provide
            any number in (0, 1, 2, 3). If left as a None, a random one will be chosen.
            Provide a number to make deterministic.

        Returns
        -------
        Evaluation
        """
        if budget is not None:
            assert 0 < budget <= self.total_budget
        else:
            budget = self.total_budget

        if index is not None:
            assert index in (0, 1, 2, 3)
        else:
            index = self.rng.choice([0, 1, 2, 3])

        # Convert to a normal dictionary as we need it as a key
        if isinstance(config, Configuration):
            config = dict(config)

        config = {
            k: (
                int(v) if isinstance(v, np.integer)
                else float(v) if isinstance(v, np.floating)
                else str(v) if isinstance(v, np.str_)
                else v
            )
            for k, v in config.items()
        }

        k = json.dumps(config, sort_keys=True)

        info = self.data[k]

        # Get the score for the algorithm with the given budget
        valid_mse = info["valid_mse"][index]
        y = valid_mse[budget - 1]

        # divide by the maximum number of epochs
        total_runtime = info["runtime"][index]
        runtime = (total_runtime / self.total_budget) * budget

        evaluation = Evaluation(y=y, runtime=runtime)

        self.evaluations.append((config, evaluation))

        return evaluation

    def learning_curve(
        self,
        config: dict | Configuration,
        budget: int | None = None,
        index: int | None = None,
    ) -> Tuple[list[float], list[float]]:
        if budget is not None:
            assert 0 < budget <= self.total_budget
        else:
            budget = self.total_budget

        # Convert to a normal dictionary as we need it as a key
        if isinstance(config, Configuration):
            config = dict(config)

        config = {
            k: (
                int(v) if isinstance(v, np.integer)
                else float(v) if isinstance(v, np.floating)
                else str(v) if isinstance(v, np.str_)
                else v
            )
            for k, v in config.items()
        }

        k = json.dumps(config, sort_keys=True)

        info = self.data[k]

        valid_mse = info["valid_mse"][index]
        total_runtime = info["runtime"][index]

        learning_curve = valid_mse[:budget]

        time_per_epoch = total_runtime / self.total_budget

        runtimes = list(accumulate(time_per_epoch for _ in range(budget)))

        return (learning_curve, runtimes)

    def test_score(self, config: dict | Configuration) -> Evaluation:
        """Get the mean test error along with it's mean runtime for a given config
        on full budget.

        Parameters
        ----------
        config : dict | Configuration
            The config to query

        Returns
        -------
        Evaluation
        """
        if isinstance(config, Configuration):
            config = dict(config)

        k = json.dumps(config, sort_keys=True)

        info = self.data[k]

        y_test = np.mean(info["final_test_error"])
        runtime = np.mean(info["runtime"])

        return Evaluation(y=y_test, runtime=runtime)

    def history(self) -> list[HistItem]:
        """Get the history of all evaluations done with the objective function

        Returns
        -------
        list[HistItem]
        """
        _, y_star_valid, y_star_test = self.best_config()

        # Get the accumulated runtime for each evaluation
        acc_runtimes = list(accumulate(ev.runtime for config, ev in self.evaluations))

        def select_lower_loss(
            prev: Tuple[dict, Evaluation],
            current: Tuple[dict, Evaluation],
        ) -> Tuple[dict, Evaluation]:
            """Returns the pair (config, Evaluation) for whichever has lower loss"""
            prev_config, prev_eval = prev
            curr_config, curr_eval = current

            if curr_eval.y < prev_eval.y:
                return (curr_config, curr_eval)
            else:
                return (prev_config, prev_eval)

        # A list of Evaluations, each one having lower loss than all the ones before it
        incumbents = list(accumulate(self.evaluations, func=select_lower_loss))

        # Get the regret with respect to the validation set
        regret_validation = [ev.y - y_star_valid for config, ev in incumbents]

        # Get the regret with respect to the test set
        regret_test = [
            self.test_score(config).y - y_star_test for config, ev in incumbents
        ]

        # Group together the history to be iterated over
        history_iter = zip(incumbents, regret_validation, regret_test, acc_runtimes)

        # Finally, stick each step in a list, creating the history
        history = [
            HistItem(
                config=config,
                y=ev.y,
                runtime=ev.runtime,
                cumulative_runtime=acc_runtime,
                regret_validation=regret_val,
                regret_test=regret_test,
            )
            for (config, ev), regret_val, regret_test, acc_runtime in history_iter
        ]

        return history

    @staticmethod
    def get_configuration_space() -> ConfigurationSpace:
        """Get the configuration space associated with these problems

        Returns
        -------
        ConfigurationSpace
            The space used in these benchmarks
        """
        cs = ConfigurationSpace()

        p1 = CategoricalHyperparameter("n_units_1", [16, 32, 64, 128, 256, 512])
        p2 = CategoricalHyperparameter("n_units_2", [16, 32, 64, 128, 256, 512])
        p3 = CategoricalHyperparameter("dropout_1", [0.0, 0.3, 0.6])
        p4 = CategoricalHyperparameter("dropout_2", [0.0, 0.3, 0.6])
        p5 = CategoricalHyperparameter("activation_fn_1", ["tanh", "relu"])
        p6 = CategoricalHyperparameter("activation_fn_2", ["tanh", "relu"])
        p7 = CategoricalHyperparameter(
            "init_lr", [5 * 1e-4, 1e-3, 5 * 1e-3, 1e-2, 5 * 1e-2, 1e-1]
        )
        p8 = CategoricalHyperparameter("lr_schedule", ["cosine", "const"])
        p9 = CategoricalHyperparameter("batch_size", [8, 16, 32, 64])

        for param in (p1, p2, p3, p4, p5, p6, p7, p8, p9):
            cs.add(param)

        return cs
