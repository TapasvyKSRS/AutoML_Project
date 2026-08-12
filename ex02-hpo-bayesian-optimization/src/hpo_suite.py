import subprocess
import sys
from hposuite import create_study
from hposuite.benchmarks import BENCHMARKS
from hposuite.optimizers import OPTIMIZERS

# TODO: Create a study with SMAC_BO, Optuna (TPE) and Scikit_Optimize Optimizers for BBOB Schwefel Function


def main():

    study = create_study(
        name="ex02_hpo_bo",
        output_dir="./hposuite",
        benchmarks=[BENCHMARKS["bbob-f20-5-0"]],
        optimizers=[
            ("SMAC_BO", {"acq_func_kwargs": {}}),
            OPTIMIZERS["Optuna"],
            ("Scikit_Optimize", {"base_estimator": "GP", "acq_func": "EI"}),
            OPTIMIZERS["NepsBO"],
        ],
        num_seeds=3,
        budget=20,
        on_error="ignore",
    )

    study.optimize()

    # TODO: Using CLI in the same python environment:
    # Execute the python command to plot the results of the study using hposuite using above arguments
    """
    python -m hposuite.plotting.incumbent_trace \
    --output_dir ./hposuite \
    --study_dir ex02_hpo_bo \
    --save_dir '../../outputs' \
    --plot_file_name hposuite-bo-comparison
    """
    subprocess.run(
        [
            sys.executable, "-m", "hposuite.plotting.incumbent_trace",
            "--output_dir", "./hposuite",
            "--study_dir", "ex02_hpo_bo",
            "--save_dir", "./outputs",
            "--plot_file_name", "hposuite-bo-comparison",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
