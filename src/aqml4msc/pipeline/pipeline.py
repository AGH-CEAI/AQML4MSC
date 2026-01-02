from typing import List

import numpy as np

from aqml4msc.logging import mlflow_utils
from aqml4msc.metrics.core import aggregate_fold_metrics, compute_classification_metrics
from aqml4msc.preprocessing.transforms import preproces_pipeline
from aqml4msc.training.base_training import BaseTraining
from aqml4msc.utils.misc import encode_labels, get_stratified_cv_splits


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
    ) -> List[dict]:
        X_source_a, X_source_b = preproces_pipeline(X)
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
                    classifier.fit(
                        train_data=train_data,
                        train_y=train_y,
                        val_data=val_data,
                        val_y=val_y,
                    )
                    preds = classifier.predict(val_data=val_data, val_y=val_y)

                    preds = label_encoder.inverse_transform(preds)
                    true_labels = label_encoder.inverse_transform(val_y)
                    metrics.append(
                        compute_classification_metrics(y_true=true_labels, y_pred=preds)
                    )
                    mlflow_utils.log_metrics(metrics[fold - 1])
                    mlflow_utils.log_classification_report(
                        y_true=true_labels, y_pred=preds
                    )
                    mlflow_utils.log_confusion_matrix(y_true=true_labels, y_pred=preds)

            aggretated_metrics = aggregate_fold_metrics(metrics)
            mlflow_utils.log_aggregated_metrics(aggretated_metrics)

        return metrics  # [pracap]
