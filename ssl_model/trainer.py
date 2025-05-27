from typing import Any, Callable

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Optimizer


class SpectralBoundaryLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_cls: Callable[..., Optimizer],
        loss_funcs: dict[str, Callable] | Callable = nn.MSELoss(),
        metric_funcs: dict[str, Callable] | None = None,
    ):
        super(SpectralBoundaryLightningModule, self).__init__()

        self.model = model
        self.save_hyperparameters(ignore=["model"])

        self.loss_funcs = loss_funcs

        self.metric_funcs = metric_funcs

        # Optimizer, scheduler
        self.optimizer_cls = optimizer_cls

        # Initialize a list to keep track of epoch-level metrics if needed
        self.train_metrics_history = (
            {key: [] for key in self.metric_funcs} if self.metric_funcs else {}
        )
        self.val_metrics_history = (
            {key: [] for key in self.metric_funcs} if self.metric_funcs else {}
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.parameters())
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        loss_dict = {}
        total_loss = 0.0
        for i, (key, loss_func) in enumerate(self.loss_funcs.items()):
            loss = loss_func(outputs)
            loss_dict[key] = loss
            total_loss += loss

        self.log(
            "train/total_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        for key, loss in loss_dict.items():
            self.log(
                f"train/{key}",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        # Metrics
        if self.metric_funcs:
            for key, metric_func in self.metric_funcs.items():
                metric = metric_func(outputs, targets)
                self.log(
                    f"train/{key}",
                    metric,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                # Optionally, store metrics for epoch_end
                self.train_metrics_history[key].append(metric)

        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        loss_dict = {}
        total_loss = 0.0
        for i, (key, loss_func) in enumerate(self.loss_funcs.items()):
            loss = loss_func(outputs)
            loss_dict[key] = loss
            total_loss += loss

        self.log(
            "val/total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        for key, loss in loss_dict.items():
            self.log(
                f"val/{key}",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        # Metrics
        if self.metric_funcs:
            for key, metric_func in self.metric_funcs.items():
                metric = metric_func(outputs, targets)
                self.log(
                    f"val/{key}",
                    metric,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                # Optionally, store metrics for epoch_end
                self.val_metrics_history[key].append(metric)

    def on_train_epoch_end(self):
        if self.metric_funcs:
            for key, metrics in self.train_metrics_history.items():
                avg_metric = torch.stack(metrics).mean()
                self.log(f"train/avg_{key}", avg_metric, prog_bar=True, logger=True)
            # Clear metrics history after logging
            self.train_metrics_history = {key: [] for key in self.metric_funcs}

    def on_validation_epoch_end(self):
        if self.metric_funcs:
            for key, metrics in self.val_metrics_history.items():
                avg_metric = torch.stack(metrics).mean()
                self.log(f"val/avg_{key}", avg_metric, prog_bar=True, logger=True)
            # Clear metrics history after logging
            self.val_metrics_history = {key: [] for key in self.metric_funcs}
