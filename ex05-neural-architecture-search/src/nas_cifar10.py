from __future__ import annotations
from pathlib import Path

import ConfigSpace
import numpy as np

from ConfigSpace import Configuration
from nasbench import api
from nasbench.lib import graph_util

from copy import deepcopy


MAX_EDGES = 9
VERTICES = 7
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NO_OP = "no-op"

AVAILABLE_OPS = {
    CONV1X1:    [1, 0, 0],
    CONV3X3:    [0, 1, 0],
    MAXPOOL3X3: [0, 0, 1],
    NO_OP:      [0, 0, 0],
}

BENCHMARK_PATH = "nasbench_only108.tfrecord"


class NASCifar10:
    def __init__(self, path: str | Path = BENCHMARK_PATH):
        """
        Initialize the NASBench dataset and the benchmark.

        Parameters
        ----------
        path: str | Path
            Path to the NASBench dataset.

        """
        self.dataset = api.NASBench(str(path))
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []

        self.y_star_valid = 0.04944576819737756  # lowest mean validation error
        self.y_star_test = 0.056824247042338016  # lowest mean test error

    def record_invalid(self, config, valid, test, costs):
        self.X.append(config)
        self.y_valid.append(valid)
        self.y_test.append(test)
        self.costs.append(costs)

    def record_valid(self, config, data, model_spec):

        self.X.append(config)

        # compute mean test error for the final budget
        _, metrics = self.dataset.get_metrics_from_spec(model_spec)
        mean_test_error = 1 - np.mean(
            [metrics[108][i]["final_test_accuracy"] for i in range(3)]
        )
        self.y_test.append(mean_test_error)

        # compute validation error for the chosen budget
        valid_error = 1 - data["validation_accuracy"]
        self.y_valid.append(valid_error)

        runtime = data["training_time"]
        self.costs.append(runtime)

    def get_results(self, ignore_invalid_configs=False):

        regret_validation = []
        regret_test = []
        runtime = []
        rt = 0

        inc_valid = np.inf
        inc_test = np.inf

        for i in range(len(self.X)):

            if ignore_invalid_configs and self.costs[i] == 0:
                continue

            if inc_valid > self.y_valid[i]:
                inc_valid = self.y_valid[i]
                inc_test = self.y_test[i]

            regret_validation.append(float(inc_valid - self.y_star_valid))
            regret_test.append(float(inc_test - self.y_star_test))
            rt += self.costs[i]
            runtime.append(float(rt))

        res = dict()
        res["regret_validation"] = regret_validation
        res["regret_test"] = regret_test
        res["runtime"] = runtime

        return res

    def get_model_spec(self, config) -> api.ModelSpec:
        """
        Convert a configuration to a model specification.

        Parameters
        ----------
        config: Configuration
            Configuration to convert to model specification.

        """
        matrix = np.zeros([VERTICES, VERTICES], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        for i in range(VERTICES * (VERTICES - 1) // 2):
            row = idx[0][i]
            col = idx[1][i]
            matrix[row, col] = config["edge_%d" % i]

        if graph_util.num_edges(matrix) > MAX_EDGES:
            self.record_invalid(config, 1, 1, 0)
            return None

        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = api.ModelSpec(matrix, labeling)

        if model_spec.valid_spec is False:
            self.record_invalid(config, 1, 1, 0)
            return None

        return model_spec

    def objective_function(self,
                           config: Configuration,
                           budget: int = 108
                           ) -> tuple[float, float]:
        """
        Compute the validation error and the runtime for a given configuration.

        Parameters
        ----------
        config: Configuration
            Configuration to evaluate.
        budget: int
            Budget for the evaluation.

        Returns
        -------
        tuple[float, float]
            Validation error and runtime.
        """
        model_spec = self.get_model_spec(config)

        if model_spec is None:
            return 1, 0

        try:
            data = self.dataset.query(model_spec, epochs=budget)
        except api.OutOfDomainError:
            self.record_invalid(config, 1, 1, 0)
            return 1, 0

        self.record_valid(config, data, model_spec)
        return 1 - data["validation_accuracy"], data["training_time"]

    @staticmethod
    def get_configuration_space():
        """
        Returns the configuration space for the NASBench dataset.
        """
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]

        for i in range(5):
            cs.add_hyperparameter(
                ConfigSpace.CategoricalHyperparameter("op_node_%d" % i, ops_choices)
            )

        for i in range(VERTICES * (VERTICES - 1) // 2):
            cs.add_hyperparameter(
                ConfigSpace.CategoricalHyperparameter("edge_%d" % i, [0, 1])
            )
        return cs

    def convert_to_vector_representation(self, model_spec: api.ModelSpec) -> np.ndarray:
        """Converts a model spec into a fixed-length vector representation.

        Parameters
        ----------
        model_spec : api.ModelSpec
            The model spec to convert to vector representation.

        Returns
        -------
        vector_representation : np.ndarray
            The fixed-length vector representation of the model spec.
        """

        matrix, ops = self.convert_to_uniform_matrix_and_ops(model_spec)
        ops = np.array([AVAILABLE_OPS[op] for op in ops[1:-1]])

        return np.concatenate((matrix.flatten(), ops.flatten()))

    def convert_to_uniform_matrix_and_ops(
            self,
            model_spec: api.ModelSpec
            ) -> tuple[np.ndarray, list[str]]:
        """Converts a model spec's matrix and ops to a fixed length (7x7 for the matrix
        and 7 for ops) representation.

        E.g., converts the model spec:
        spec.matrix = [
            [0, 1, 1], # input node
            [0, 0, 1], # intermediate node
            [0, 0, 0]  # output node
        ]

        spec.ops = ['input', 'conv3x3-bn-relu', 'output']

        into the following fixed length representation:

        matrix = [
            [0, 1, 0, 0, 0, 0, 1], # input node
            [0, 0, 0, 0, 0, 0, 1], # first intermediate node
            [0, 0, 0, 0, 0, 0, 0], # this and the next three rows indicate nodes which
            [0, 0, 0, 0, 0, 0, 0], # are not present in this model spec
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0], # output node
        ]

        ops = [
            'input',
            'conv3x3-bn-relu',
            'no-op',
            'no-op',
            'no-op',
            'no-op',
            'output'
        ]

        Parameters
        ----------
        model_spec : api.ModelSpec
            The model spec to convert to a fixed length representation.

        Returns
        -------
        matrix : np.ndarray
            The fixed-dimension matrix representation of the model spec.
        ops : list[str]
            The list of ops in the fixed-dimension representation of the model spec.
        """
        matrix = model_spec.matrix
        ops = model_spec.ops

        n_nodes = matrix.shape[0]

        new_ops = deepcopy(ops)

        for _ in range(VERTICES - n_nodes):
            new_ops.insert(-1, "no-op")

        new_matrix = np.zeros([VERTICES, VERTICES])
        new_matrix[:n_nodes, :n_nodes] = matrix

        for idx, row in enumerate(new_matrix):
            if row[n_nodes - 1] == 1:
                row[n_nodes - 1] = 0
                row[-1] = 1

            if idx == n_nodes:
                break

        return new_matrix, new_ops
