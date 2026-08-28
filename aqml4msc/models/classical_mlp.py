import torch.nn as nn


import torch

def classical_2l_mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim[0]),
        nn.ReLU(),
        nn.Linear(hidden_dim[0], output_dim),
    )


class ConcatMLPFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list[int], output_dim: int):
        super().__init__()
        self.network = classical_2l_mlp(input_dim, hidden_dim, output_dim)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        fused = torch.cat(features, dim=-1)
        return self.network(fused)
