from hposuite import create_study
from hposuite.plotting.incumbent_trace import agg_data

if __name__ == "__main__":

    # TODO: Use the hposuite library to create a study to compare RandomSearch and CMA-ES for the ackley benchmark
    study = create_study(
        name=...,
        optimizers=["RandomSearch", ("Nevergrad", {"optimizer_name": "CMA-ES"})],
        benchmarks=...,
        budget=...,
        num_seeds=...,
        output_dir=...,
    )

    study.optimize()

    # TODO: Using CLI in the same python environment:
    # Execute the python command to plot the results of the study using hposuite using above arguments
