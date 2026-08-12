from typing import List, Tuple
import numpy as np

from pytest import approx

from collections import Counter
from src.hpo import determine_best_hypers
from src.evolution import Mutation, ParentSelection, Recombination


def test_hpo() -> None:
    """Test resulting best configuration when overfitting to the ackley function

    Expects
    -------
    * Recombination.INTERMEDIATE to be chosen
    * Recombination.INTERMEDIATE should be chosen more than all other recombination
        methods combined
    """
    # We simply test if we always come close to the optimum
    all_configs: List[Tuple[Mutation, ParentSelection, Recombination]] = []
    all_perfs: List[float] = []

    # we evaluate over multiple seeds to get a better estimate of the true performance
    for i in range(30):
        config, perf = determine_best_hypers()

        # Configs have to be tuples in the following order (Mutation, Selection, Recombination)
        all_configs.append(config)
        all_perfs.append(perf)

    assert np.mean(all_perfs) == approx(0.0, abs=1)

    # We only care about the Recombination strategy here
    recombination_strategies = [config[2] for config in all_configs]

    # By nature of the ackley function the Intermediate recombination strategy
    # is expected to be much more preferable than any other recombination strategy

    counts = Counter(recombination_strategies)

    assert Recombination.INTERMEDIATE in counts

    # Should be chosen more than the sum of times any other choice was picked
    other_choices = [Recombination.NONE, Recombination.UNIFORM]
    assert counts[Recombination.INTERMEDIATE] > sum(counts[c] for c in other_choices)
