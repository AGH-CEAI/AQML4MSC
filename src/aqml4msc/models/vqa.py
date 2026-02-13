from typing import List

import pennylane as qml
import torch
from torch import nn

from aqml4msc.models.base_mlp_model import BaseMLPModel
from aqml4msc.utils.misc import probe_inputs_and_weight_shapes


class QMLP_1(BaseMLPModel):
    def __init__(
        self,
        lr: float,
        loss_fn,
        num_classes: int,
        input_dim: int,
        hidden_dim_part: List[int],
        n_qubits: int,
        n_layers: int,
        device_name: str = "default.qubit",
    ):
        super().__init__(lr=lr, loss_fn=loss_fn, num_classes=num_classes)

        # Set output_dim_part to be equal in both parts and match the n_qubits — which is the input size for angle encoding
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=n_qubits)
        output_dim_part = n_qubits // 2
        self.model_top = self.make_classical_network(
            input_dim=input_dim, hidden_dim=hidden_dim_part, output_dim=output_dim_part
        )
        self.model_bottom = self.make_classical_network(
            input_dim=input_dim, hidden_dim=hidden_dim_part, output_dim=output_dim_part
        )
        self.model_classifier = self.make_quantum_classifier(
            n_layers=n_layers, num_classes=num_classes
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

    def make_quantum_classifier(self, num_classes, n_layers=3):
        @qml.qnode(self.dev)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_classes)]

        weight_shapes = {"weights": (n_layers, self.n_qubits)}

        return qml.qnn.TorchLayer(qnode, weight_shapes)  # type: ignore

    def apply_ansatz(self, circuit):
        qnode = qml.QNode(circuit, self.dev)
        weight_shapes = probe_inputs_and_weight_shapes(qnode)
        weight_shapes.pop("inputs", None)
        print(weight_shapes)
        qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)  # type: ignore
        self.model_classifier = qlayer
