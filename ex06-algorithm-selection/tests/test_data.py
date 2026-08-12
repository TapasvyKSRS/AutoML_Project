from pathlib import Path

import arff

HERE = Path(__file__).parent.resolve()
DATADIR = HERE.parent / "data"
SAT11_INDU = DATADIR / "SAT11-INDU" / "algorithm_runs.arff"
SAT11_RAND = DATADIR / "SAT11-RAND" / "algorithm_runs.arff"


def data_individual_models(tricky: bool = False):
    data = [
        ["a", 1.0, "A", 1.0, "ok"],
        ["a", 1.0, "B", 3.0, "ok"],

        ["b", 1.0, "A", 2.0, "ok"],
        ["b", 1.0, "B", 1.0, "ok"],

        ["c", 1.0, "A", 3.0, "ok"],
        ["c", 1.0, "B", 1.0, "ok"],

        ["d", 1.0, "A", 1.0, "ok"],
        ["d", 1.0, "B", 3.0, "ok"],

        ["e", 1.0, "A", 1.0, "ok"],
        ["e", 1.0, "B", 4.0, "timeout"],

        ["f", 1.0, "A", 5.0, "timeout"],
        ["f", 1.0, "B", 3.0, "ok"],
    ]

    features = [["a", 0], ["b", 1], ["c", 1], ["d", 0], ["e", 0], ["f", 1]]
    features_tricky = [["a", 0], ["b", 1], ["c", 1], ["d", 0], ["e", 1], ["f", 0]]

    cv = [["a", 1, 1], ["b", 1, 1], ["c", 1, 2], ["d", 1, 2], ["e", 1, 3], ["f", 1, 3]]

    return_features = features_tricky if tricky else features

    return data, return_features, cv


def data_hybrid_models():
    data = [
        ["a", 1.0, "A", 1.0, "ok"],
        ["a", 1.0, "B", 3.0, "ok"],
        ["a", 1.0, "C", 4.0, "timeout"],

        ["b", 1.0, "A", 2.0, "ok"],
        ["b", 1.0, "B", 1.0, "ok"],
        ["b", 1.0, "C", 4.0, "timeout"],

        ["c", 1.0, "A", 3.0, "ok"],
        ["c", 1.0, "B", 1.0, "ok"],
        ["c", 1.0, "C", 4.0, "timeout"],

        ["d", 1.0, "A", 1.0, "ok"],
        ["d", 1.0, "B", 3.0, "ok"],
        ["d", 1.0, "C", 4.0, "timeout"],

        ["e", 1.0, "A", 1.0, "ok"],
        ["e", 1.0, "B", 4.0, "timeout"],
        ["e", 1.0, "C", 0.0, "ok"],

        ["f", 1.0, "A", 5.0, "timeout"],
        ["f", 1.0, "B", 3.0, "ok"],
        ["f", 1.0, "C", 0.0, "ok"],
    ]

    features = [["a", 0], ["b", 1], ["c", 1], ["d", 0], ["e", 2], ["f", 2]]
    cv = [["a", 1, 1], ["b", 1, 1], ["c", 1, 2], ["d", 1, 2], ["e", 1, 3], ["f", 1, 3]]

    return data, features, cv


def data_simple_stats() -> list:
    data = [
        ["a", 1.0, "A", 1.0, "ok"],
        ["a", 1.0, "B", 3.0, "ok"],

        ["b", 1.0, "A", 2.0, "ok"],
        ["b", 1.0, "B", 1.0, "ok"],

        ["c", 1.0, "A", 3.0, "ok"],
        ["c", 1.0, "B", 1.0, "ok"],

        ["d", 1.0, "A", 1.0, "ok"],
        ["d", 1.0, "B", 3.0, "ok"],

        ["e", 1.0, "A", 1.0, "ok"],
        ["e", 1.0, "B", 4.0, "timeout"],
        # even though a 5 is recorded as runtime value,
        # the penalized value, will be 4*2 instead of 5*2
        ["f", 1.0, "A", 5.0, "timeout"],
        ["f", 1.0, "B", 3.0, "ok"],
    ]
    # Oracle    = 1+1+1+1+1+3   =  8/6 = 1.333333
    # A         = 1+2+3+1+1+4*2 = 16/2 = 2.666666
    # B         = 3+1+1+3+4*2+3 = 19/6 = 3.166666
    return data


def data_real_stats(kind: str) -> list:
    assert kind.lower() in ["indu", "rand"]
    if kind.lower() == "indu":
        path = SAT11_INDU
    else:
        path = SAT11_RAND

    with path.open("r") as f:
        d = arff.load(f)
        return d["data"]
