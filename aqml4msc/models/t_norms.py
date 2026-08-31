import torch


def _validate_and_stack(features: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(features)
    if torch.any(stacked < 0.0) or torch.any(stacked > 1.0):
        raise ValueError("All input values for t-norms must be in the range [0, 1]")
    return stacked


def product_t_norm(features: list[torch.Tensor]) -> torch.Tensor:
    return torch.prod(_validate_and_stack(features), dim=0)


def godel_t_norm(features: list[torch.Tensor]) -> torch.Tensor:
    return torch.amin(_validate_and_stack(features), dim=0)


def lukasiewicz_t_norm(features: list[torch.Tensor]) -> torch.Tensor:
    return torch.clamp(
        torch.sum(_validate_and_stack(features), dim=0) - (len(features) - 1), min=0
    )
