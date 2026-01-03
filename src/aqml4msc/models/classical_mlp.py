import torch
import torch.nn as nn

from aqml4msc.models.base_mlp_model import BaseMLPModel


class CMLP_1(BaseMLPModel):
    def __init__(
        self,
        input_dim,
        hidden_dim_part,
        output_dim_part,
        hidden_dim_class,
        num_classes,
        loss_fn,
        lr=1e-3,
    ):
        super().__init__(lr=lr, loss_fn=loss_fn, num_classes=num_classes)
        self.model_top = self.make_classical_network(
            input_dim=input_dim, hidden_dim=hidden_dim_part, output_dim=output_dim_part
        )
        self.model_bottom = self.make_classical_network(
            input_dim=input_dim, hidden_dim=hidden_dim_part, output_dim=output_dim_part
        )
        self.model_classifier = self.make_classical_classifier(
            input_dim=2 * output_dim_part,
            hidden_dim=hidden_dim_class,
            num_classes=num_classes,
        )

    def forward(self, x_top, x_bottom):
        features1 = self.model_top(x_top)
        features2 = self.model_bottom(x_bottom)
        logits = self.model_classifier(torch.cat([features1, features2], dim=1))
        return logits

    def make_classical_network(self, input_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], output_dim),
        )

    def make_classical_classifier(self, input_dim, hidden_dim, num_classes):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], num_classes),
        )
