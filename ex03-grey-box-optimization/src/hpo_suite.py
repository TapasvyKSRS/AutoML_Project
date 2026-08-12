from hposuite import create_study
from hposuite.benchmarks import BENCHMARKS
from hposuite.optimizers import OPTIMIZERS

# TODO: Create a study
# with your desired optimizers and benchmarks.

study = create_study(
    name="ex03_grey_box",  # name of the study
    budget=20,  # explore the budget space
    num_seeds=3,  # number of seeds to run
    output_dir="./hposuite",
    optimizers=[
        OPTIMIZERS["NepsSuccessiveHalving"],
        OPTIMIZERS["NepsHyperband"],
        OPTIMIZERS["SMAC_Hyperband"],
    ],
    benchmarks=[
        BENCHMARKS["mfh3_good"],
    ],
)
study.optimize()  # starts the optimization process
# TODO: Using CLI in the same python environment:

"""
python -m hposuite.plotting.incumbent_trace \
--output_dir ./hposuite \
--study_dir ex03_grey_box \
--save_dir '../../outputs' \
--plot_file_name hposuite-bo-comparison \
--benchmark_spec {BENCHMARK_NAME}
"""
