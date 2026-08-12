from __future__ import annotations

from itertools import accumulate

from ConfigSpace import Configuration, ConfigurationSpace, OrdinalHyperparameter

from src.problem import Evaluation, HistItem, Problem


class ToyBenchmark(Problem):
    def __init__(self, seed: int):
        super().__init__(seed)

    def objective_function(
        self,
        config: dict | Configuration,
        budget: int,
    ) -> Evaluation:
        """Pretend to evaluate the configuration's value and runtime

        Parameters
        ----------
        config: dict | Configuration
            The configuration to "evaluate"

        budget: float
            The budget in second for which to evaluate this config for

        Returns
        -------
        Evaluation
            The config, y and runtime of this configuration
        """
        y = self.rng.uniform(0.1, 3.0) / budget
        runtime = self.rng.uniform(0.001, 0.01) * budget

        evaluation = Evaluation(y=y, runtime=runtime)

        self.evaluations.append((config, evaluation))

        return evaluation

    def history(self) -> list[HistItem]:
        """Get the result of the evaluations

        Returns
        -------
        list[HistItem]
        """

        # Keeps the minimum as it iterates through [7, 5, 10, 3] -> [7, 5, 5, 3]
        regret_scores = accumulate((ev.y for config, ev in self.evaluations), min)

        # Add all the runtimes [1,2,3] -> [1, 3, 6]
        runtimes = accumulate(ev.runtime for config, ev in self.evaluations)

        # Create an iterator that will go through the history
        history_iter = zip(self.evaluations, regret_scores, runtimes)

        history = [
            HistItem(
                config=config,
                y=ev.y,
                runtime=ev.runtime,
                cumulative_runtime=acc_runtime,
                regret_validation=regret,  # type: ignore
            )
            for (config, ev), regret, acc_runtime in history_iter
        ]

        return history

    @classmethod
    def get_configuration_space(cls) -> ConfigurationSpace:
        """Get the configuration space for this problem

        Returns
        -------
        ConfigurationSpace
            The config space for this problem
        """
        cs = ConfigurationSpace()

        p1 = OrdinalHyperparameter("hyp_1", [16, 32, 64, 128, 256, 512])
        p2 = OrdinalHyperparameter("hyp_2", [16, 32, 64, 128, 256, 512])

        cs.add_hyperparameter(p1)
        cs.add_hyperparameter(p2)

        return cs
