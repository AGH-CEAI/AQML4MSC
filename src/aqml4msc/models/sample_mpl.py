import torch.nn as nn

from aqml4msc.models.base_mlp_model import BaseMLPModel


class SimpleMLP(BaseMLPModel):
    def __init__(self, input_dim, num_classes, lr=1e-3):
        super().__init__(lr)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )
