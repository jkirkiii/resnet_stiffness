#
#
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class StiffnessLoss(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def foo(self, block, block_inputs):
        jacobians = torch.autograd.functional.jacobian(block, block_inputs)
        pass

    def analysis_eigenvalues(jacobians):
        eigen_values, eigen_vectors = torch.linalg.eig(jacobians)
        return eigen_values

    def test(self, input):
        return input / 2

