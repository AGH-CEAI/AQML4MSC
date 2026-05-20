from typing import Any, Callable, Tuple

import pytorch_lightning as pl
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class BaseMLPModel(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        extractor_factories: list[Callable[[], torch.nn.Module]],
        fusion_factory: Callable[[], torch.nn.Module],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn"])

        # Instantiate sub-models by calling the partial functions
        self.extractors = torch.nn.ModuleList(
            [factory() for factory in extractor_factories]
        )
        self.fusion = fusion_factory()

        # TorchMetrics
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=model_params["num_classes"]),
                "f1": MulticlassF1Score(num_classes=model_params["num_classes"]),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.loss_fn = model_params["loss_fn"]
        self.lr = model_params["lr"]

    def forward(self, *sources) -> torch.Tensor:
        # Pass each source through its respective extractor
        features = [extractor(src) for extractor, src in zip(self.extractors, sources)]
        # Fusion model needs fusion logic implemented in its forward method
        return self.fusion(features)

    def training_step(self, batch, batch_idx):
        *inputs, labels = batch
        logits = self(*inputs)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)

        # log loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        # log metrics (handles update/compute internally)
        self.log_dict(
            self.train_metrics(preds, labels),
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        *inputs, labels = batch
        logits = self(*inputs)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        self.log_dict(
            self.val_metrics(preds, labels),
            on_epoch=True,
            prog_bar=True,
        )

    def predict_step(self, batch: Tuple[Any, ...], batch_idx: int) -> torch.Tensor:
        logits = self(*batch)
        return torch.argmax(logits, dim=1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)  # type: ignore
