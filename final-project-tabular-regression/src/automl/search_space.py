"""
Search space definitions for each candidate model family.

Bounds are justified in the report:
  - Ranges cover both under- and over-fitting regimes while excluding clearly
    wasteful settings (e.g. GBM learning rates above ~0.3 tend to diverge on
    datasets of a few thousand rows).
  - Log scale is used for parameters spanning several orders of magnitude
    (learning rates, regularisation strengths) so the optimiser samples evenly
    across magnitudes rather than being biased toward large values.

The families deliberately span distinct inductive biases:
  linear / regularised   : ridge, elasticnet, bayesian_ridge
  bagging ensembles      : random_forest, extra_trees
  boosting ensembles     : lightgbm, hist_gradient_boosting, catboost
  pretrained / in-context: tabpfn
"""
from __future__ import annotations

from ConfigSpace import (
    ConfigurationSpace,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
)

from automl.models import AVAILABLE_MODELS


# ----------------------------- linear / regularised -------------------------

def ridge_space(seed: int = 0) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        UniformFloatHyperparameter("alpha", 1e-4, 100.0, default_value=1.0, log=True),
    ])
    return cs


def elasticnet_space(seed: int = 0) -> ConfigurationSpace:
    """L1+L2. l1_ratio near 0 behaves like Ridge, near 1 like Lasso; keeping the
    interior open lets the optimiser choose how much feature selection to do."""
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        UniformFloatHyperparameter("alpha", 1e-4, 10.0, default_value=1.0, log=True),
        UniformFloatHyperparameter("l1_ratio", 0.01, 1.0, default_value=0.5),
    ])
    return cs


def bayesian_ridge_space(seed: int = 0) -> ConfigurationSpace:
    """BayesianRidge infers its own regularisation strength, so we only expose
    the Gamma hyperpriors. Wide log ranges centred on the defaults (1e-6)."""
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        UniformFloatHyperparameter("alpha_1", 1e-8, 1e-2, default_value=1e-6, log=True),
        UniformFloatHyperparameter("alpha_2", 1e-8, 1e-2, default_value=1e-6, log=True),
        UniformFloatHyperparameter("lambda_1", 1e-8, 1e-2, default_value=1e-6, log=True),
        UniformFloatHyperparameter("lambda_2", 1e-8, 1e-2, default_value=1e-6, log=True),
    ])
    return cs


# ----------------------------- bagging ensembles ----------------------------

def random_forest_space(seed: int = 0) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("n_estimators", 50, 800, default_value=200, log=True),
        UniformIntegerHyperparameter("max_depth", 3, 30, default_value=15),
        UniformIntegerHyperparameter("min_samples_split", 2, 20, default_value=2),
        UniformIntegerHyperparameter("min_samples_leaf", 1, 20, default_value=1),
        UniformFloatHyperparameter("max_features", 0.3, 1.0, default_value=1.0),
    ])
    return cs


def extra_trees_space(seed: int = 0) -> ConfigurationSpace:
    """Same knobs as RandomForest; the difference is algorithmic (random split
    thresholds rather than optimised ones), not in the search space."""
    return random_forest_space(seed)


# ---------------------------- boosting ensembles ----------------------------

def lightgbm_space(seed: int = 0) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("n_estimators", 50, 1000, default_value=200, log=True),
        UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.05, log=True),
        UniformIntegerHyperparameter("max_depth", 3, 12, default_value=6),
        UniformIntegerHyperparameter("num_leaves", 15, 255, default_value=31),
        UniformIntegerHyperparameter("min_child_samples", 5, 100, default_value=20),
        UniformFloatHyperparameter("subsample", 0.5, 1.0, default_value=1.0),
        UniformFloatHyperparameter("colsample_bytree", 0.5, 1.0, default_value=1.0),
        UniformFloatHyperparameter("reg_alpha", 1e-8, 10.0, default_value=1e-8, log=True),
        UniformFloatHyperparameter("reg_lambda", 1e-8, 10.0, default_value=1e-8, log=True),
    ])
    return cs


def hist_gradient_boosting_space(seed: int = 0) -> ConfigurationSpace:
    """sklearn's histogram GBM. Uses max_iter rather than n_estimators and has no
    column subsampling, so the space genuinely differs from LightGBM's."""
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("max_iter", 50, 1000, default_value=200, log=True),
        UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True),
        UniformIntegerHyperparameter("max_leaf_nodes", 15, 255, default_value=31),
        UniformIntegerHyperparameter("min_samples_leaf", 5, 100, default_value=20),
        UniformFloatHyperparameter("l2_regularization", 1e-8, 10.0, default_value=1e-8, log=True),
        UniformIntegerHyperparameter("max_bins", 64, 255, default_value=255),
    ])
    return cs


def catboost_space(seed: int = 0) -> ConfigurationSpace:
    """CatBoost uses ordered boosting and ordered target statistics, which tend
    to help on categorical-heavy data (yprop_4_1 has 20 categorical columns)."""
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        UniformIntegerHyperparameter("iterations", 50, 1000, default_value=200, log=True),
        UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.05, log=True),
        UniformIntegerHyperparameter("depth", 3, 10, default_value=6),
        UniformFloatHyperparameter("l2_leaf_reg", 1e-2, 30.0, default_value=3.0, log=True),
        UniformFloatHyperparameter("random_strength", 1e-3, 10.0, default_value=1.0, log=True),
    ])
    return cs


# --------------------------- pretrained / in-context ------------------------

def tabpfn_space(seed: int = 0) -> ConfigurationSpace:
    """TabPFN is a pretrained transformer doing in-context learning: there are no
    weights to train and essentially nothing to tune. The only meaningful knob is
    how many ensemble members to average, trading accuracy against runtime.

    Included as a nearly hyperparameter-free family so the screening stage can
    compare a foundation model against models trained from scratch.
    """
    cs = ConfigurationSpace(seed=seed)
    cs.add_hyperparameters([
        CategoricalHyperparameter("n_estimators", [1, 2, 4, 8], default_value=4),
    ])
    return cs


# ------------------------------- registry -----------------------------------

_ALL_SPACES = {
    "ridge": ridge_space,
    "elasticnet": elasticnet_space,
    "bayesian_ridge": bayesian_ridge_space,
    "random_forest": random_forest_space,
    "extra_trees": extra_trees_space,
    "lightgbm": lightgbm_space,
    "hist_gradient_boosting": hist_gradient_boosting_space,
    "catboost": catboost_space,
    "tabpfn": tabpfn_space,
}
SEARCH_SPACES = {k: v for k, v in _ALL_SPACES.items() if k in AVAILABLE_MODELS}