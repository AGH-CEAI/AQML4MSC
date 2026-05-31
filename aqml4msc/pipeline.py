from functools import partial

import aqml4msc.logging as logging
from aqml4msc.datasets.base_dataset import BaseDataset
from aqml4msc.metrics import aggregate_fold_metrics
from aqml4msc.training.base_training import BaseTraining
from aqml4msc.utils import get_stratified_cv_splits, set_seeds


class ClassificationPipeline:
    def process_data(
        self,
        model_factory: partial,
        dataset: BaseDataset,
        training: BaseTraining,
        params: dict,
    ) -> dict:
        set_seeds(params["experiment_params"]["seed"])

        dataset.load_raw()
        dataset.prepare_data()

        metrics = []
        preds = []
        true_labels = []

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
                    dataset.preprocess()

                    model = model_factory()

                    training.fit(model, dataset)
                    preds_encoded = training.predict(model, dataset)

                    preds.append(dataset.decode_labels(preds_encoded))
                    true_labels.append(dataset.decode_labels(dataset.val_labels))

                    metrics = logging.log_all_run_metrics(
                        metrics,
                        true_labels[-1],
                        preds[-1],
                        dataset,
                        fold,
                        training,
                        model,
                        model_name=params["experiment_params"]["model_name"],
                    )

            aggretated_metrics = aggregate_fold_metrics(metrics)
            logging.log_aggregated_metrics(aggretated_metrics, preds, true_labels)

        return aggretated_metrics  # [pracap]
