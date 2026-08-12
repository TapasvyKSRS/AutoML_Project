import os
import sys 
import unittest
import numpy as np
import torch
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from src.model_search import Network
from src.train_search import Architect


class TestDARTS(unittest.TestCase):
    def test_ex1_b_nas(self):
        device = torch.device("cpu")

        # Seed the environment
        np.random.seed(0)
        torch.manual_seed(0)
        model = Network(device, nodes=2).to(device)

        architect = Architect(model)

        input_valid = torch.randn(size=(1, 1, 28, 28), dtype=torch.float32).to(device)
        input_target = torch.zeros(size=(1, 1), dtype=torch.long).reshape(-1).to(device)
        # Perform one-step
        architect.step(input_valid, input_target)
        output_agg = torch.sum(model.arch_parameters[0]).detach().numpy()
        print(output_agg)
        self.assertTrue(np.isclose(output_agg, -0.0050634164, rtol=1.e-2))


if __name__ == '__main__':
    unittest.main()
