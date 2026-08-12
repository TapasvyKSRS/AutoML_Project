from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import rcParams

rcParams.update({"figure.autolayout": True})
plt.style.use("seaborn-v0_8-whitegrid")


def plot_grey_box_optimization(
    configs_list: list[dict] | dict,
    min_budget_per_model: int,
    kind: str,
) -> Figure:
    output_dir = "./outputs/"
    os.makedirs(output_dir, exist_ok=True)
    if kind == "successive-halving":
        assert isinstance
        n_rows, n_cols = 1, 1
        filename = "successive_halving_results.pdf"
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(5, 5), sharex="col", sharey="row"
        )
        axs = [axs]

    elif kind == "hyperband":
        n_hyperband_iter = len(configs_list)
        n_cols = int(
            (n_hyperband_iter - (n_hyperband_iter % 3)) / 3 + (n_hyperband_iter % 3)
        )
        n_rows = 3
        filename = "hyperband_results.pdf"
        fig, axs = plt.subplots(
            3, n_cols, figsize=(n_cols * n_rows, n_rows * 2), sharex="col", sharey="row"
        )
        axs = axs.reshape(-1)

    else:
        raise NotImplementedError(kind)

    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for idx, (configs_dict, ax) in enumerate(zip(configs_list, axs)):
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.3)

        all_budgets = []
        for config_id, (config, evals) in configs_dict.items():
            budgets = list(evals.keys())
            val_errors = [eval.y for eval in evals.values()]

            ax.scatter(budgets, val_errors, s=6)
            ax.plot(budgets, val_errors)

            all_budgets.extend(budgets)

        # Use the same x-axis limits for all subplots for easier comparison.
        if min_budget_per_model > 1:
            ax.set_xlim(min_budget_per_model - 1, 110)
        else:
            ax.set_xlim(min_budget_per_model, 110)

        for budget in np.unique(all_budgets):
            ax.axvline(budget, c="black", lw=0.5)

        if idx == 0:
            ax.set_xlabel("Budget (Epochs)")
            ax.set_ylabel("Validation Error")

    plt.tight_layout()
    plt.savefig(output_dir + filename)
    return fig
