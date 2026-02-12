"""
This module implements AQML approach for ansatz finding using :mod:`aqmlator` package.

.. note::

    Bases heavily on ``src.aqml4msc.experiments``.
    TODO(SD): Consider refactoring the code, so that both are using single implementation of the
    common part of the code.
"""

from aqmlator.qml import AnsatzBuilder
from aqmlator.tuner import AnsatzFinder
from numpy.typing import NDArray
import numpy as np
import optuna
import pennylane as qml

from qmetric.quantum_circuit_metrics import (
    quantum_locality_ratio,
    effective_entanglement_entropy,
    quantum_mutual_information,
)

from pennylane_qiskit.converter import circuit_to_qiskit
from qiskit.quantum_info import Statevector

from torch import nn

from aqml4msc.logging.mlflow_utils import EpochMetricsTracker
from aqml4msc.models.vqa import QMLP_1
from aqml4msc.training.mlp_training import MLPTraining
from statistics import mean

# TODO(SD): Module :mod:`aqml4msc.data.loading`, which I copied from `src.aqml4msc.experiments.quantum_hpo`
#           Does not exsist. Adjust it adequately.
from aqml4msc.data.loading import choose_digits, load_data
from aqml4msc.pipeline.pipeline import ClassificationPipeline
import qiskit
from typing import Any


def suggest_ansatz(
    trial: optuna.Trial,
) -> callable:  # TODO(TR): Typehint this properly.
    """
    Use AQML methods implemented in :mod:`aqmlator` to suggest, build and return an ansatz.

    .. important::
        Ansatze suggested by :mod:`aqmlator` are without measurements! Remember to add them
        prior to use!

    :param trial: _description_
    :type trial: optuna.Trial

    :return:
        Ansatz suggested by the :mod:`aqmlator`.
    :rtype: callable
    """
    ansatz_finder: AnsatzFinder = AnsatzFinder(
        n_wires=trial.params["n_qubits"],
        n_min_blocks=trial.params["n_layers"],
        n_max_blocks=trial.params["n_layers"],
    )

    ansatz_recipe: dict[str, Any] = ansatz_finder.suggest_ansatz(trial)

    return AnsatzBuilder.from_recipe(ansatz_recipe)


def get_vqc_metrics(
    pennylane_circuit: qml.QNode, n_qubits: int, model_weights: NDArray[np.floating]
) -> dict[str, float]:
    """Compute selected metrics (quantum locality ratio, effective entanglement entropy, quantum mutual information)
    for the given ``pennylane_circuit``, and given ``n_qubits`` and ``weights``. In the process, the circuit will
    be transformed into ``qiskit.QuantumCircuit`` so that :mod:`qmetric` can be used.

    :param pennylane_circuit:
        A :class:`pennylane.QNode` representing the VQC, for which the metrics are to be computed.
    :type pennylane_circuit: qml.QNode
    :param n_qubits:
        The number of qubits used in ``pennylane_circuit``. Perhaps redundant.
    :type n_qubits: int
    :param model_weights:
        Trained weights of the ``pennylane_circuit``.
    :type model_weights: NDarray[np.floating]

    :return:
        The values of selected QC metrics in the form of a :class:`dict`.
    :rtype: dict[str, float]
    """
    metrics: dict[str, float] = {}

    tape: qml.tape.QuantumScript = qml.tape.make_qscript(pennylane_circuit)(
        n_qubits, np.array(weights)
    )
    qiskit_cirtuit: qiskit.QuantumCircuit = circuit_to_qiskit(tape, n_qubits)
    qiskit_cirtuit.remove_final_measurements()

    final_state = Statevector.from_instruction(qiskit_cirtuit)
    subsystem_a: list[int] = list(range(n_qubits // 2))
    subsystem_b: list[int] = list(range(n_qubits // 2, n_qubits))

    metrics["qlr"] = float(quantum_locality_ratio(qiskit_cirtuit))
    metrics["eee"] = float(
        effective_entanglement_entropy(final_state, subsystem_qubits=subsystem_a)
    )
    metrics["qmi"] = float(
        quantum_mutual_information(final_state, subsystem_a, subsystem_b)
    )

    return metrics


def optuna_aqml_objective(trial: optuna.Trial) -> float:
    """ """
    model_params = {
        "lr": 1e-3,
        "loss_fn": nn.CrossEntropyLoss(),
        "num_classes": 3,
        "input_dim": 14,
        "hidden_dim_part": [trial.suggest_int("hidden_dim_part", 64, 256)],
        "n_qubits": trial.suggest_int("n_qubits", low=4, high=16, step=2),
        "n_layers": trial.suggest_int("n_layers", low=1, high=5),
    }

    trainer_params = {
        "max_epochs": 30,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "num_sanity_val_steps": 0,
        "callbacks": [EpochMetricsTracker()],
        "logger": False,
        "accelerator": "auto",
        "devices": "auto",
    }

    data_params = {
        "batch_size": trial.suggest_int("batch_size", 32, 128),
        "num_workers": 8,
        "digits": [5, 6, 7],
    }

    experiment_params = {
        "seed": 42,
        "n_folds": 5,
        "parent_run_name": "QMLP_HPO_1",
        "model_name": "QMLP_1",
    }

    training = MLPTraining(
        model_cls=QMLP_1,
        model_kwargs=model_params,
        trainer_kwargs=trainer_params,
        batch_size=data_params["batch_size"],
    )

    X, y = load_data()
    X, y = choose_digits(X, y, data_params["digits"])
    pipeline = ClassificationPipeline()

    # TODO(SD):  Refactor the code, so that quantum circuit can be applied to the model PRIOR to the training.
    ansatz: callable = suggest_ansatz(trial)

    def circuit(inputs, weights) -> list[float]:
        ansatz(inputs, weights)
        return [
            qml.expval(qml.PauliZ(wires=i)) for i in range(model_params["num_classes"])
        ]

    

    # TODO(SD): The object that you want to put in the model is ``circuit``.
    training.model.model_classifier.apply_ansatz(circuit)  # Weights!

    metrics = pipeline.process_data(
        X=X,
        y=y,
        classifier=training,
        experiment_params=experiment_params,
        data_params=data_params,
        model_params=model_params,
        trainer_params=trainer_params,
        # Albo
        # ansatz=ansatz,
    )

    # TODO(SD): Once trained, if you could extract the VQC and it's weights, from the ``trainer`` or the ``pipeline``
    #           you can compute the quantum circuit metrics (qcm) that you wanted, like so:
    # vqc = pipeline.model
    # weights = vqc.weights
    # qcm: dict[str, float] = get_vqc_metrics(vqc, model_params["n_qubits"], vqc_weights)
    # TODO(SD) Remeber to store them somewhere!

    # TODO(SD): You may consider using more classification metrics, and multiobjective optimization.
    return mean(metrics["accuracy"])


def main() -> None:
    """Calls the experiment."""
    study: optuna.Study = optuna.create_study(direction="maximize")
    study.optimize(optuna_aqml_objective, n_trials=20)
    print(study.best_params)


if __name__ == "__main__":
    print("Experiment start")
    main()
    print("Experiment finished")
