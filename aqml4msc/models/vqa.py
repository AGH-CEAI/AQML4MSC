from typing import Callable

import numpy as np
import pennylane as qml
import torch
from pennylane.devices import Device
from pennylane.qnn import TorchLayer
from qmetric.converters import PennylaneToQASM3


def ansatz_to_torch(
    ansatz: Callable,
    dev: Device,
    n_qubits_measured: int,
    weight_shapes: dict[str, tuple[int, ...]] | None = None,
) -> TorchLayer:
    if weight_shapes is None:

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


def quantum_linear(torch_layer: TorchLayer, n_qubits_measured: int, num_output: int):
    return torch.nn.Sequential(
        torch_layer,  # type: ignore
        torch.nn.ReLU(),
        torch.nn.Linear(n_qubits_measured, num_output),
    )


class ConcatVQAFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        fused = torch.cat(features, dim=-1)
        return self.network(fused)


class VQAAnsatz(ConcatVQAFusion):
    def __init__(
        self,
        dev: Device,
        num_classes: int,
        ansatz: Callable | None = None,
        weight_shapes: dict[str, tuple[int, ...]] | None = None,
    ):
        super().__init__()
        self.ansatz = ansatz
        self.network = ansatz_to_torch(ansatz, dev, num_classes, weight_shapes)


class VQAAnsatzLinear(ConcatVQAFusion):
    def __init__(
        self,
        dev: Device,
        n_qubits_measured: int,
        num_classes: int,
        ansatz: Callable | None = None,
        weight_shapes: dict[str, tuple[int, ...]] | None = None,
    ):
        super().__init__()
        self.ansatz = ansatz
        self.torch_layer = ansatz_to_torch(
            ansatz, dev, n_qubits_measured, weight_shapes
        )
        self.network = quantum_linear(self.torch_layer, n_qubits_measured, num_classes)


# --- ANSATZE ---
def ansatz_angle_basic(n_qubits: int) -> Callable:
    def ansatz(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

    return ansatz


def ansatz_amplitude_basic(n_qubits: int) -> Callable:
    def ansatz(inputs, weights):
        qml.AmplitudeEmbedding(
            inputs, wires=range(n_qubits), pad_with=0, normalize=True
        )
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

    return ansatz


def ansatz_amplitude_strongly(n_qubits: int) -> Callable:
    def ansatz(inputs, weights):
        qml.AmplitudeEmbedding(
            inputs, wires=range(n_qubits), pad_with=0, normalize=True
        )
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return ansatz


def ansatz_amplitude_seperate_strongly(
    n_qubits: list[int], feat_sizes: list[int]
) -> Callable:
    def ansatz(inputs, weights):
        it_qubits = 0
        it_features = 0
        for i, n in enumerate(n_qubits):
            qml.AmplitudeEmbedding(
                inputs[..., it_features : it_features + feat_sizes[i]],
                wires=range(it_qubits, it_qubits + n),
                pad_with=0,
                normalize=True,
            )
            it_qubits += n
            it_features += feat_sizes[i]
        qml.StronglyEntanglingLayers(weights, wires=range(sum(n_qubits)))

    return ansatz


# --- CUSTOM ---
class VQAAmplStrongProbs(ConcatVQAFusion):
    def __init__(self, dev: Device, num_classes: int, n_qubits: int):
        super().__init__()
        ansatz = ansatz_amplitude_strongly(n_qubits)
        weight_shapes = {"weights": (1, n_qubits, 3)}

        @qml.qnode(dev)
        def circuit(inputs, weights):  # type: ignore
            ansatz(torch.as_tensor(inputs), torch.as_tensor(weights))
            return qml.probs(wires=range(n_qubits))

        self.network = quantum_linear(
            TorchLayer(circuit, weight_shapes), 2**n_qubits, num_classes
        )


class VQAAmplStrongProbs_Bilinear_Pool(ConcatVQAFusion):
    def __init__(
        self, dev: Device, num_classes: int, n_qubits: list[int], feat_sizes: list[int]
    ):
        super().__init__()
        ansatz = ansatz_amplitude_seperate_strongly(n_qubits, feat_sizes)
        weight_shapes = {"weights": (1, sum(n_qubits), 3)}

        @qml.transforms.broadcast_expand
        @qml.qnode(dev)
        def circuit(inputs, weights):  # type: ignore
            ansatz(torch.as_tensor(inputs), torch.as_tensor(weights))
            return qml.probs(wires=range(sum(n_qubits)))

        self.network = quantum_linear(
            TorchLayer(circuit, weight_shapes), 2 ** sum(n_qubits), num_classes
        )
