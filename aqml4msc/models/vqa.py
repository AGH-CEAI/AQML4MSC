from typing import Callable

import numpy as np
import pennylane as qml
import torch
from pennylane.devices import Device
from pennylane.qnn import TorchLayer
from qmetric.converters import PennylaneToQASM3


def ansatz_to_torch(
    ansatz: Callable, dev: Device, n_qubits_measured: int
) -> TorchLayer:
    @qml.qnode(dev)
    def probe_circuit(inputs, weights) -> list[qml.measurements.ExpectationMP]:  # type: ignore
        ansatz(np.array(inputs), np.array(weights))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits_measured)]

    weight_shapes = PennylaneToQASM3.probe_inputs_and_weight_shapes(probe_circuit)
    weight_shapes.pop("inputs")

    @qml.qnode(dev)
    def circuit(inputs, weights) -> list[qml.measurements.ExpectationMP]:  # type: ignore
        ansatz(torch.as_tensor(inputs), torch.as_tensor(weights))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits_measured)]

    return TorchLayer(circuit, weight_shapes)  # type: ignore


def ansatz_angle_basic(n_qubits: int) -> Callable:
    def ansatz(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

    return ansatz


def quantum_linear(torch_layer: TorchLayer, n_qubits_measured: int, num_output: int):
    return torch.nn.Sequential(
        torch_layer,  # type: ignore
        torch.nn.ReLU(),
        torch.nn.Linear(n_qubits_measured, num_output),
    )


class ConcatVQAFusion(torch.nn.Module):
    def __init__(self, dev: Device, num_classes: int, ansatz: Callable | None = None):
        super().__init__()
        self.ansatz = ansatz
        self.network = ansatz_to_torch(ansatz, dev, num_classes)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        fused = torch.cat(features, dim=-1)
        return self.network(fused)
