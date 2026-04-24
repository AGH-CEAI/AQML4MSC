import numpy as np

import aqml4msc.logging as logging
from aqml4msc.metrics import aggregate_fold_metrics
from aqml4msc.preprocessing.mnist import preprocess_pipeline
from aqml4msc.training.base_training import BaseTraining
from aqml4msc.utils import encode_labels, get_stratified_cv_splits, set_seeds


class ClassificationPipeline:
    def process_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifier: BaseTraining,
        params: dict,
        ansatz=None,  # TODO(SD) To refactor
    ) -> dict:
        model_params = params['model_params']
        trainer_params = params['trainer_params']
        data_params = params['data_params']
        experiment_params = params['experiment_params']
        optuna_params = params.get('optuna_params', {})

        set_seeds(experiment_params["seed"])

        X_source_a, X_source_b = preprocess_pipeline(X)
        label_encoder, y = encode_labels(y)

        logging.setup_mlflow()
        metrics = []

        with logging.start_parent_run(model_name=experiment_params["parent_run_name"]):
            logging.log_params(model_params)
            logging.log_params(trainer_params)
            logging.log_params(data_params)
            logging.log_params(experiment_params)
            logging.log_params(optuna_params)

            for fold, train_idx, val_idx in get_stratified_cv_splits(
                y=y,
                n_folds=experiment_params["n_folds"],
                start_idx=1,
                seed=experiment_params["seed"],
            ):
                with logging.start_child_hp_run(f"Fold {fold}"):
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

                    metrics = logging.log_all_run_metrics(
                        metrics,
                        true_labels,
                        preds,
                        val_data,
                        fold,
                        classifier,
                        model_name=experiment_params["model_name"],
                        ansatz=ansatz,
                    )

            aggretated_metrics = aggregate_fold_metrics(metrics)
            logging.log_aggregated_metrics(aggretated_metrics)

        return aggretated_metrics  # [pracap]
