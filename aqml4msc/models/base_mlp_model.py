from typing import Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class BaseMLPModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        num_classes: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn"])

        # self.model = ... to be implemented in subclasses

        # TorchMetrics
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "f1": MulticlassF1Score(num_classes=num_classes),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.loss_fn = loss_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)  # type: ignore
