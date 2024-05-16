import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureSoftmax(nn.Module):
    def __init__(self, temperature=1, dim=None):
        super().__init__()
        self.temperature = temperature
        self.dim = dim

    def forward(self, input):
        return F.softmax(input / self.temperature, dim=self.dim)

class SplineMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, control_points):
        super(SplineMLP, self).__init__()

        self.control_points = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim * control_points)
        )

        self.n = control_points
        self.output_dim = output_dim
        self.basis_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, control_points),
            TemperatureSoftmax(0.5, dim=1)  # normalize the output to make it a proper basis function
        )

    def forward(self, x):
        # bs, control points
        basis_values = self.basis_network(x)  # learnable basis function
        outputs = self.control_points(x).reshape(-1, self.n, self.output_dim)
        outputs = torch.einsum('bno,bn->bo', outputs, basis_values)
        return outputs