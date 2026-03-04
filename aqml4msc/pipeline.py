import numpy as np
import pennylane as qml
from aqmlator.tuner import compute_qc_metrics

import aqml4msc.mlflow_utils as mlflow_utils
from aqml4msc.metrics import aggregate_fold_metrics, compute_classification_metrics
from aqml4msc.preprocessing import preprocess_pipeline
from aqml4msc.training.base_training import BaseTraining
from aqml4msc.utils import encode_labels, get_stratified_cv_splits, set_seeds


class ClassificationPipeline:
    def process_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifier: BaseTraining,
        model_params: dict,
        trainer_params: dict,
        data_params: dict,
        experiment_params: dict,
        optuna_params: dict = {},
        ansatz=None,  # TODO(SD) To refactor
    ) -> dict:
        set_seeds(experiment_params["seed"])

        X_source_a, X_source_b = preprocess_pipeline(X)
        label_encoder, y = encode_labels(y)

        mlflow_utils.setup_mlflow()
        metrics = []

        with mlflow_utils.start_parent_run(
            model_name=experiment_params["parent_run_name"]
        ):
            mlflow_utils.log_params(model_params)
            mlflow_utils.log_params(trainer_params)
            mlflow_utils.log_params(data_params)
            mlflow_utils.log_params(experiment_params)
            mlflow_utils.log_params(optuna_params)

            for fold, train_idx, val_idx in get_stratified_cv_splits(
                y=y,
                n_folds=experiment_params["n_folds"],
                start_idx=1,
                seed=experiment_params["seed"],
            ):
                with mlflow_utils.start_child_hp_run(f"Fold {fold}"):
                    train_data = (X_source_a[train_idx], X_source_b[train_idx])
                    train_y = y[train_idx]
                    val_data = (X_source_a[val_idx], X_source_b[val_idx])
                    val_y = y[val_idx]

                    classifier.reset_model()

                    if ansatz is not None:
                        classifier.model.apply_ansatz(ansatz)

                    classifier.fit(
                        train_data=train_data,
                        train_y=train_y,
                        val_data=val_data,
                        val_y=val_y,
                    )
                    preds = classifier.predict(val_data=val_data)

                    preds = label_encoder.inverse_transform(preds)
                    true_labels = label_encoder.inverse_transform(val_y)

                    # TODO(SD): Separete method for metrics logging.
                    metrics.append(
                        compute_classification_metrics(y_true=true_labels, y_pred=preds)
                    )

                    metrics[-1].update(
                        compute_qc_metrics(
                            qml.QNode(ansatz, device=classifier.model.dev)
                        )
                    )
                    mlflow_utils.log_metrics(metrics[fold - 1])
                    try:
                        mlflow_utils.log_classification_report(
                            y_true=true_labels, y_pred=preds
                        )
                        mlflow_utils.log_confusion_matrix(
                            y_true=true_labels, y_pred=preds
                        )
                        mlflow_utils.log_model(
                            trainer=classifier,
                            X_val=val_data,
                            model_name=experiment_params["model_name"],
                        )
                    except Exception as e:
                        print(f"Could not save artifacts. Error occured: {e}")

            aggretated_metrics = aggregate_fold_metrics(metrics)
            mlflow_utils.log_aggregated_metrics(aggretated_metrics)

        return aggretated_metrics  # [pracap]
