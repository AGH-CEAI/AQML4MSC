from datasets.base_dataset import BaseDataset

import aqml4msc.logging as logging
from aqml4msc.metrics import aggregate_fold_metrics
from aqml4msc.training.base_training import BaseTraining
from aqml4msc.utils import get_stratified_cv_splits, set_seeds


class ClassificationPipeline:
    def process_data(
        self,
        dataset: BaseDataset,
        training: BaseTraining,
        params: dict,
        ansatz=None,  # TODO(SD) To refactor
    ) -> dict:
        set_seeds(params["experiment_params"]["seed"])

        dataset.load_raw()
        dataset.clean_data()
        dataset.preprocess()

        metrics = []

        with logging.start_parent_run(
            model_name=params["experiment_params"]["parent_run_name"]
        ):
            logging.log_params(params)

            for fold, train_idx, val_idx in get_stratified_cv_splits(
                y=dataset.get_encoded_labels(),
                n_folds=params["experiment_params"]["n_folds"],
                start_idx=1,
                seed=params["experiment_params"]["seed"],
            ):
                with logging.start_child_hp_run(f"Fold {fold}"):
                    dataset.set_splits(train_idx, val_idx)

                    training.reset_model()

                    if ansatz is not None:
                        training.model.apply_ansatz(ansatz)

                    training.fit(dataset=dataset)
                    preds = training.predict(val_data=dataset.val_data)

                    preds = dataset.decode_labels(preds)
                    true_labels = dataset.decode_labels(dataset.val_labels)

                    metrics = logging.log_all_run_metrics(
                        metrics,
                        true_labels,
                        preds,
                        dataset.val_data,
                        fold,
                        training,
                        model_name=params["experiment_params"]["model_name"],
                        ansatz=ansatz,
                    )

            aggretated_metrics = aggregate_fold_metrics(metrics)
            logging.log_aggregated_metrics(aggretated_metrics)

        return aggretated_metrics  # [pracap]
