#
#
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class StiffnessLoss:
    def __init__(self, rate):
        self.rate = rate

    def calculate_stiffness(self, model, inputs):
        jacobians = torch.autograd.functional.jacobian(model, inputs)
        eigen_values = self.get_eigenvalues(jacobians)
        eigen_max = torch.max(eigen_values)
        eigen_max *= self.rate
        return eigen_max

    def get_eigenvalues(self, jacobians):
        eigen_values, eigen_vectors = torch.linalg.eig(jacobians)
        return eigen_values
