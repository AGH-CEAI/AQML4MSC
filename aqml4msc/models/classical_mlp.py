import torch
import torch.nn as nn


def classical_2l_mlp(
    input_dim: int, hidden_dim: list[int], output_dim: int
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim[0]),
        nn.ReLU(),
        nn.Linear(hidden_dim[0], output_dim),
    )


def classical_1l_sigmoid_mlp(input_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.Sigmoid(),
    )


class ConcatMLPFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list[int], output_dim: int):
        super().__init__()
        self.network = classical_2l_mlp(input_dim, hidden_dim, output_dim)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        fused = torch.cat(features, dim=-1)
        return self.network(fused)


class TNormMLPFusion(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: list[int], output_dim: int, t_norm: callable
    ):
        super().__init__()
        self.network = classical_2l_mlp(input_dim, hidden_dim, output_dim)
        self.t_norm = t_norm

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        fused = self.t_norm(features)
        return self.network(fused)
