import os
import arff
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# TODO import a regressor from sklearn to solve task 2
# TODO import a classifier from sklearn to solve task 3


def get_stats(
    aslib_data: List,
    cutoff: float,
    par: float = 10.0,
) -> Tuple[float, float, str]:
    """Determine Virtual best and Single best performance.

    Expects input data in the ASLib data format.

    Parameters
    ----------
    aslib_data : List
        Entries are formatted as:
            [
                ('instance_id', 'STRING'),
                ('repetition', 'NUMERIC'),
                ('algorithm', 'STRING'),
                ('runtime', 'NUMERIC'),
                ('runstatus', ['ok', 'timeout', 'memout', 'not_applicable', 'crash', 'other'])
            ]

    cutoff : float
        The cutoff for the algorithm

    par : float = 10
        The penalization factor.
        In the case of an error or timeout, then (runtime = PAR * cutoff)

    Returns
    -------
    Tuple[float, float, str]
        The oracle_performance, single_best_performance, and single_best_algo_name

    Notes
    -----
    The input is a list of per-run records, not a matrix. The outputs are
    scalar means over instances (one value for oracle and one for single best).
    When a run is a timeout or error, the runtime should be treated as
    (par * cutoff).
    If multiple repetitions exist for the same instance/algorithm, each
    repetition is treated as its own run for the single-best mean, while the
    oracle takes the minimum over all runs per instance.
    """
    # pandas data frames allow for easy data handling
    df = pd.DataFrame(
        aslib_data,
        columns=[
            "instance_id",
            "repetition",
            "algorithm",
            "runtime",
            "runstatus",
        ],
    )

    # Default dict docs https://docs.python.org/3/library/collections.html#defaultdict-examples
    algos = defaultdict(list)  # track individual algorithm performances
    insts = defaultdict(lambda: np.inf)  # track individual instance performances

    oracle_perf, single_best_perf, single_best_algo = 0.0, 0.0, ""

    # TODO determine the oracle and single best performance.
    # Your function must also determine and return the name (string)
    # of the algorithm that achieved the Single Best performance.
    # ---------------------
    # Penalize non-ok runs with par * cutoff
    df["penalized"] = np.where(df["runstatus"] == "ok", df["runtime"], par * cutoff)

    # Oracle: for each instance take min runtime over all algorithms/repetitions, then average
    oracle_perf = float(df.groupby("instance_id")["penalized"].min().mean())

    # Single best: algorithm with the lowest mean penalized runtime over all (instance, rep) pairs
    algo_means = df.groupby("algorithm")["penalized"].mean()
    single_best_algo = str(algo_means.idxmin())
    single_best_perf = float(algo_means.min())
    # ---------------------

    return oracle_perf, single_best_perf, single_best_algo


def hybrid_model(
    test_instances: List[str],
    algos: List[str],
    run_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    test_feature_df: pd.DataFrame,
) -> List[int]:
    """Use pairwise classification to predict which algorithm will outperform the others.

    Based on that, build your selection.

    Parameters
    ----------
    test_instances : List[str]
        List of instance ids

    algos : List[str]
        List of algorithms

    run_df : pd.DataFrame
        All the training runtime data (i.e. y_train)

    feature_df : pd.DataFrame
        All training feature data (i.e. X_train)

    test_feature_df : pd.DataFrame
        All test feature_data (i.e. X_test)

    Returns
    -------
    List[int]
        List of selected algorithms per test instance. I.e index of element in `algos`
        that should be used to solve an instance in test_instances

    Notes
    -----
    X_train = feature_df.values[:, 1:] with shape (n_train_instances, n_features)
    X_test = test_feature_df.values[:, 1:] with shape (n_test_instances, n_features)
    For each (outer, inner) algorithm pair, build y_train as a 1D array of length
    n_train_instances where 1 means outer is faster than inner for that instance.
    """
    y_predictions = np.zeros((len(test_instances), len(algos)))
    X_train = feature_df.values[:, 1:]
    X_test = test_feature_df.values[:, 1:]

    instance_ids = feature_df["instance_id"].values

    # Pre-compute mean runtime per instance for each algorithm (handles multiple repetitions)
    algo_runtimes = {}
    for algo in algos:
        rt = run_df[run_df["algorithm"] == algo].groupby("instance_id")["runtime"].mean()
        algo_runtimes[algo] = rt.reindex(instance_ids).fillna(rt.mean()).values

    # TODO for each pair of algorithms, fit a model that classifies if outer outperforms inner.
    # Use voting to decide which algorithm solves which instance
    for idx_outer, algo_outer in enumerate(algos):
        for idx_inner, algo_inner in enumerate(algos):
            if algo_outer == algo_inner:
                continue
            y_train = (algo_runtimes[algo_outer] < algo_runtimes[algo_inner]).astype(int)
            clf = RandomForestClassifier(n_estimators=10, random_state=42)
            clf.fit(X_train, y_train)
            y_predictions[:, idx_outer] += clf.predict(X_test)

    selection = y_predictions.argmax(axis=1)
    return selection


def individual_model(
    test_instances: List[str],
    algos: List[str],
    run_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    test_feature_df: pd.DataFrame,
) -> List[int]:
    """

    Parameters
    ----------
    test_instances : List[str]
        List of instance ids

    algos : List[str]
        List of algorithms

    run_df : pd.DataFrame
        All the training runtime data (i.e. y_train)

    feature_df : pd.DataFrame
        All training feature data (i.e. X_train)

    test_feature_df : pd.DataFrame
        All test feature_data (i.e. X_test)

    Returns
    -------
    List[int]
        The selected algorithms per test instance. I.e index of element in `algos` that
        should be used to solve an instance in test_instances.

    Notes
    -----
    X_train = feature_df.values[:, 1:] with shape (n_train_instances, n_features)
    X_test = test_feature_df.values[:, 1:] with shape (n_test_instances, n_features)
    For each algorithm, y_train is a 1D array of runtimes aligned to X_train
    (one value per training instance). Predicted y_test should be length
    n_test_instances.
    """
    y_predictions = np.zeros((len(test_instances), len(algos)))
    X_train = feature_df.values[:, 1:]
    X_test = test_feature_df.values[:, 1:]

    instance_ids = feature_df["instance_id"].values

    # TODO for each algorithm, fit a model and predict the performance on the test instances
    for idx, algo in enumerate(algos):
        rt = run_df[run_df["algorithm"] == algo].groupby("instance_id")["runtime"].mean()
        y_train = rt.reindex(instance_ids).fillna(rt.mean()).values
        reg = RandomForestRegressor(n_estimators=10, random_state=42)
        reg.fit(X_train, y_train)
        y_predictions[:, idx] = reg.predict(X_test)

    selection = y_predictions.argmin(axis=1)
    return selection


def select(
    aslib_run_data: List,
    aslib_feature_data: List,
    aslib_cv_splits: List,
    cutoff: int,
    parx: float,
    algos: Optional[List] = None,
    test_split_index: Optional[int] = 10,
    individual: bool = True,
) -> Tuple[float, List[int]]:
    """Trains individual regression models for each algorithm and predicts the
    performance on the final cv split.

    Parameters
    ----------
    aslib_run_data : List
        List of ASlib run data. Entries are as follows:
        [
            ('instance_id', 'STRING'),
            ('repetition', 'NUMERIC'),
            ('algorithm', 'STRING'),
            ('runtime', 'NUMERIC'),
            ('runstatus', ['ok', 'timeout', 'memout', 'not_applicable', 'crash', 'other'])
        ]

    aslib_feature_data : List
        List of ASlib feature data.
        [
            ('instance_id', 'STRING'),
            ('feature_1', 'NUMERIC'),
            ('feature_2', 'NUMERIC'),
            ...,
            ('feature_N', 'NUMERIC')
        ]

    aslib_cv_splits : List
        ASlib cv splits. Entries are as follows:
        [
            ('instance_id', 'STRING'),
            ('repetitions', 'NUMERIC'),
            ('split_id', 'NUMERIC')
        ]

    cutoff : int
         the maximum allowed runtime value

    parx : float
        the penalization factor for timed-out runs.

    algos : Optional[List] = None
        List of algorithms to consider. If using the ASLib data is too expensive with
        all algorithms, you can specify (as a list of strings) which algorithms you
        want to consider to speed up computation. If set to None, all algorithms are considered

    test_split_index : Optional[int] = 10
        Which CV index is left out for evaluation purposes

    individual : bool = True
        Boolean to determine if the individual method or the hybrid method should be used.

    Returns
    -------
    Tuple[float, List[int]]
        The mean runtime over test instances and the selected algorithm indices.
        The selection list has length n_test_instances.
    """
    # pandas data frames allow for easy data handling
    run_df = pd.DataFrame(aslib_run_data)

    # correctly name the entries
    run_df.columns = ["instance_id", "repetition", "algorithm", "runtime", "runstatus"]

    # replace timeouts with penalized runtime
    run_df["runtime"] = run_df["runtime"].apply(
        lambda runtime: runtime if runtime < cutoff else cutoff * parx
    )

    feature_df = pd.DataFrame(aslib_feature_data)
    cols = ["instance_id"]
    for i in range(feature_df.shape[1] - 1):
        cols.append("feat_%d" % i)

    feature_df.columns = cols

    cv_df = pd.DataFrame(aslib_cv_splits)
    cv_df.columns = ["instance_id", "repetition", "split"]

    if test_split_index:
        assert 0 < test_split_index < 11, "Only values from 1-10 are valid"

        # determine train and test instances
        train_instances = cv_df[cv_df["split"] != test_split_index][
            "instance_id"
        ].values
        test_instances = cv_df[cv_df["split"] == test_split_index]["instance_id"].values

    else:
        # `train` and `test` the same used for test purposes
        test_instances = train_instances = cv_df["instance_id"].unique()

    if not algos:
        algos = run_df["algorithm"].unique()  # all algorithms we need to fit models for
    else:
        run_df = run_df[run_df["algorithm"].isin(algos)]

    assert algos is not None

    # we sort all entries according to the instances such that
    # we have an easy match from run-data to feature-data
    test_run_df = run_df[run_df["instance_id"].isin(test_instances)].sort_values(
        ["instance_id", "algorithm"]
    )

    run_df = run_df[run_df["instance_id"].isin(train_instances)].sort_values(
        ["instance_id", "algorithm"]
    )

    test_feature_df = feature_df[
        feature_df["instance_id"].isin(test_instances)
    ].sort_values("instance_id")

    feature_df = feature_df[
        feature_df["instance_id"].isin(train_instances)
    ].sort_values("instance_id")

    # impute missing feature values (nan values) with mean values
    test_feature_df = test_feature_df.fillna(feature_df.mean(numeric_only=True))
    feature_df = feature_df.fillna(feature_df.mean(numeric_only=True))

    # Complete the following two methods
    # * individual_model
    # * hybrid_model
    if individual:
        selection = individual_model(
            test_instances=test_instances,
            algos=algos,
            run_df=run_df,
            feature_df=feature_df,
            test_feature_df=test_feature_df,
        )
    else:
        selection = hybrid_model(
            test_instances=test_instances,
            algos=algos,
            run_df=run_df,
            feature_df=feature_df,
            test_feature_df=test_feature_df,
        )

    performance = []
    for instance, sel in zip(test_instances, selection):
        row = test_run_df[
            (test_run_df["algorithm"] == algos[sel])
            & (test_run_df["instance_id"] == instance)
        ]
        performance.append(row["runtime"].iloc[0])

    mean = np.mean(performance)

    return mean, selection


def full_eval():
    """
    Evaluates the Single Best, Virtual Best, Individual Model, and Hybrid Model
    on the SAT11-INDU and SAT11-RAND datasets.
    Outputs the results as a Markdown table using pandas.
    """
    print("Starting full evaluation on ASLib data...")
    BASE_DIR = Path(__file__).parent.parent
    scenarios = ["SAT11-INDU", "SAT11-RAND"]
    results = []

    for scenario in scenarios:
        print(f"Evaluating {scenario}...")
        data_dir = BASE_DIR / "data" / scenario

        with open(data_dir / "algorithm_runs.arff", "r") as f:
            run_data = arff.load(f)["data"]
        with open(data_dir / "feature_values.arff", "r") as f:
            feature_data = arff.load(f)["data"]
        with open(data_dir / "cv.arff", "r") as f:
            cv_data = arff.load(f)["data"]

        cv_df = pd.DataFrame(cv_data, columns=["instance_id", "repetition", "split"])
        test_instances = cv_df[cv_df["split"] == 10]["instance_id"].values
        train_instances = cv_df[cv_df["split"] != 10]["instance_id"].values

        run_df = pd.DataFrame(run_data, columns=["instance_id", "repetition", "algorithm", "runtime", "runstatus"])

        test_run_df = run_df[run_df["instance_id"].isin(test_instances)].copy()
        train_run_df = run_df[run_df["instance_id"].isin(train_instances)].copy()

        test_run_data = test_run_df.values.tolist()
        train_run_data = train_run_df.values.tolist()

        # ---------------------------------------------------------
        # TODO: Compute Virtual Best (VB)
        # Use `get_stats` on `test_run_data` to get the VB.
        # Set cutoff=5000 and par=10.
        vb_perf, _, _ = get_stats(test_run_data, cutoff=5000, par=10)

        # TODO: Compute Single Best (SB) strictly using Train -> Test evaluation
        # 1. Use `get_stats` on `train_run_data` to extract the winning algorithm's name.
        # 2. Calculate the PAR10 penalized runtime of that specific algorithm on the `test_run_df`.
        # (Hint: use np.where(test_run_df["runstatus"] == "ok", test_run_df["runtime"], 5000 * 10))
        _, _, sb_algo = get_stats(train_run_data, cutoff=5000, par=10)
        sb_test = test_run_df[test_run_df["algorithm"] == sb_algo].copy()
        sb_perf = float(
            np.where(sb_test["runstatus"] == "ok", sb_test["runtime"], 5000 * 10).mean()
        )

        # TODO: Compute Individual Model performance
        # Use `select` with the FULL data lists (run_data, feature_data, cv_data).
        # Set cutoff=5000, parx=10, test_split_index=10, and individual=True.
        print(f"  Running individual model for {scenario}...")
        ind_perf, _ = select(
            run_data, feature_data, cv_data,
            cutoff=5000, parx=10, test_split_index=10, individual=True
        )

        # TODO: Compute Hybrid Model performance
        # Use `select` with the FULL data lists (run_data, feature_data, cv_data).
        # Set cutoff=5000, parx=10, test_split_index=10, and individual=False.
        print(f"  Running hybrid model for {scenario}...")
        hyp_perf, _ = select(
            run_data, feature_data, cv_data,
            cutoff=5000, parx=10, test_split_index=10, individual=False
        )
        # ---------------------------------------------------------

        results.append({
            "Scenario": scenario,
            "Virtual Best (VB)": round(vb_perf, 1),
            "Single Best (SB)": round(sb_perf, 1),
            "Individual Model": round(ind_perf, 1),
            "Hybrid Model": round(hyp_perf, 1)
        })

    os.makedirs(BASE_DIR / "outputs", exist_ok=True)
    out_path = BASE_DIR / "outputs" / "evaluation.md"

    df = pd.DataFrame(results)

    with open(out_path, "w") as f:
        f.write("# ASLib Evaluation Results\n\n")
        f.write(df.to_markdown(index=False))

    print(f"\nEvaluation complete! Results saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ASLib models and baselines.")
    parser.add_argument("--eval_all", action="store_true",
                        help="Run the full evaluation pipeline and generate markdown.")
    args = parser.parse_args()

    if args.eval_all:
        full_eval()
