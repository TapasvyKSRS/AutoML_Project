from pytest import mark, param, approx

from src.aslib import get_stats, select
from tests.test_data import (data_hybrid_models, data_individual_models,
                             data_real_stats, data_simple_stats)


# If you'd like to add a test case just include one in the list and make sure you provide
# a valud for each parameter specified in `mark.parametrize`. You can ignore the `id=""`
# or just provide your own one. The ids we use are needed for the grading and should
# not be used by you.
@mark.parametrize(
    "data, features, cv, is_individual",
    [
        param(*data_hybrid_models(), False, id="hybrid"),
        param(*data_individual_models(), True, id="individual"),
        param(*data_individual_models(tricky=True), True, id="individual_tricky"),
    ],
)
def test_select(data: list, features: list, cv: list, is_individual: bool, request) -> None:
    m, selection = select(
        aslib_run_data=data,
        aslib_feature_data=features,
        aslib_cv_splits=cv,
        cutoff=4,
        parx=2,
        algos=None,
        test_split_index=None,
        individual=is_individual,
    )
    o, s, _ = get_stats(data, cutoff=4, par=2)

    assert o <= m <= s, (
        f"Expected oracle <= mean <= single best; got oracle={o}, mean={m}, single_best={s}"
    )

    # NOTE: We don't test this for the "individual_tricky" version, old note:
    #
    #   Due to the change of instance features for instances of e and f we can't simply
    #   test the correct selection like in the prior test
    #
    if request.node.callspec.id == "individual_tricky":
        return

    else:
        # selection should be perfectly matched to feature
        print("features", features)
        print("selection", selection)
        for feature, sel in zip(features, selection):
            assert feature[1] == sel, (
                f"Selection mismatch for instance {feature[0]}: expected {feature[1]}, got {sel}"
            )


@mark.parametrize(
    "data, cutoff, par, expected_oracle_perf, expected_single_best_perf, expected_sb_algo",
    [
        param([["a", 1.0, "A", 1.0, "ok"]], 100, 10, 1.0, 1.0, "A", id="sample"),
        param(data_simple_stats(), 4, 2, 1.333, 2.667, "A", id="sample"),
        param(data_real_stats("indu"), 5_000, 10, 8187.5, 14605.9, "glucose_2", id="real"),
        param(data_real_stats("rand"), 5_000, 10, 9186.4, 19916.4, "sparrow2011_sparrow2011_ubcsat1.2_2011-03-02", id="real"),
    ],
)
def test_stats(
    data: list,
    cutoff: int,
    par: float,
    expected_oracle_perf: float,
    expected_single_best_perf: float,
    expected_sb_algo: str,
) -> None:
    oracle_perf, single_best_perf, sb_algo = get_stats(
        data,
        cutoff=cutoff,
        par=par,
    )
    
    assert oracle_perf == approx(expected_oracle_perf, abs=1e-1), \
        f"Virtual Best (Oracle) performance mismatch. Expected {expected_oracle_perf:.3f}, but got {oracle_perf:.3f}."
        
    assert single_best_perf == approx(expected_single_best_perf, abs=1e-1), \
        f"Single Best performance mismatch. Expected {expected_single_best_perf:.3f}, but got {single_best_perf:.3f}."
        
    assert sb_algo == expected_sb_algo, \
        f"Single Best algorithm mismatch. Expected '{expected_sb_algo}', but got '{sb_algo}'."
