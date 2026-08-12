from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent.resolve()
LOGPATH = HERE.parent / "outputs" / "logs"
RESULTS_PATH = LOGPATH / "results.obj"


@pytest.fixture()
def results() -> dict[str, np.ndarray]:
    if not RESULTS_PATH.exists():
        raise RuntimeError(
            f"Expected the results to be stored at {RESULTS_PATH}."
            "\nIf you haven't generated them yet, please do so by running `run.py`."
            " If you have please make sure they can be found at the given path!"
        )
    with RESULTS_PATH.open("rb") as f:
        return pickle.load(f)


def test_re(results: dict[str, np.ndarray]) -> None:
    """
    Expects
    -------
    * The mean result should be with one std. deviation of the expected value
    """
    re_results = np.array(results["re"])
    final_mean = np.mean(re_results[:, -1])

    expected = 0.05954026579856873
    tol = 0.003533369563780998
    assert (expected - tol) < final_mean < (expected + tol)


def test_rs(results: dict[str, np.ndarray]) -> None:
    """
    Expects
    -------
    * The mean result should be with one std. deviation of the expected value
    """
    rs_results = np.array(results["rs"])
    final_mean = np.mean(rs_results[:, -1])

    expected = 0.05908454656600952
    tol = 0.0020750128579245424
    assert (expected - tol) < final_mean < (expected + tol)
