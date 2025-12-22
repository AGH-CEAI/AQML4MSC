from typing import List

import pennylane as qml
import torch
from torch import nn

from src.models.base_mlp_model import BaseMLPModel


class QMLP_1(BaseMLPModel):
    def __init__(
        self,
        lr: float,
        loss_fn,
        num_classes: int,
        input_dim: int,
        hidden_dim: List[int],
        n_qubits: int,
        n_layers: int,
    ):
        super().__init__(lr=lr, loss_fn=loss_fn, num_classes=num_classes)
        self.model_top = self.make_classical_network(
            input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes
        )
        self.model_bottom = self.make_classical_network(
            input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes
        )
        self.model_classifier = self.make_quantum_classifier(
            n_qubits=n_qubits, n_layers=n_layers
        )

    def forward(self, x_top, x_bottom):
        features1 = self.model_top(x_top)
        features2 = self.model_bottom(x_bottom)
        logits = self.model_classifier(torch.cat([features1, features2], dim=1))
        return logits

    def make_classical_network(self, input_dim, hidden_dim, num_classes):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], num_classes),
        )

    def make_quantum_classifier(self, n_qubits=6, n_layers=3):
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(3)]

        weight_shapes = {"weights": (n_layers, n_qubits)}

        return qml.qnn.TorchLayer(qnode, weight_shapes)  # type: ignore
