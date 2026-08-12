import json
from pathlib import Path

import numpy as np

from src.toy_benchmark import ToyBenchmark
from src.hyperband import hyperband
from src.successive_halving import successive_halving

HERE = Path(__file__).parent.resolve()
DATADIR = HERE / "data"


def test_successive_halving():
    problem = ToyBenchmark(seed=0)
    _ = successive_halving(
        problem=problem,
        n_models=40,
        eta=2,
        random_seed=0,
        max_budget_per_model=100,
        min_budget_per_model=10,
    )

    # Get the history of the runs
    history = problem.history()
    regret_validation = [step.regret_validation for step in history]

    # Get the expected results
    path = DATADIR / "test_sh_data.json"
    with path.open("r") as f:
        expected = json.load(f)

    np.testing.assert_allclose(regret_validation, expected, atol=0.05)


def test_hyperband():
    problem = ToyBenchmark(seed=0)
    _ = hyperband(
        problem=problem,
        eta=2,
        random_seed=0,
        max_budget_per_model=100,
        min_budget_per_model=2,
    )

    # Get the history of the runs
    history = problem.history()
    regret_validation = [step.regret_validation for step in history]

    # Get the expected results
    path = DATADIR / "test_hb_data.json"
    with path.open("r") as f:
        expected = json.load(f)

    np.testing.assert_allclose(regret_validation, expected, atol=0.05)
