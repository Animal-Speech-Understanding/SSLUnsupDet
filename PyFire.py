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
        optimizer_params: dict[str, Any],
        scheduler_cls: Callable[..., Any] | None = None,
        scheduler_params: dict[str, Any] | None = None,
        loss_funcs: dict[str, Callable] | Callable = nn.MSELoss(),
        metric_funcs: dict[str, Callable] | None = None,
        multi_loss_weights: list | None = None,
        regularizer: dict[str, Any] | None = None,
        weights_func: Callable[[list, int], list] | None = None,
        switcher: dict[str, Any] | None = None,
    ):
        super(SpectralBoundaryLightningModule, self).__init__()

        self.model = model
        self.save_hyperparameters(ignore=["model"])

        # Handle loss functions
        if isinstance(loss_funcs, dict):
            self.loss_funcs = loss_funcs
        elif callable(loss_funcs):
            self.loss_funcs = {"Loss": loss_funcs}
        else:
            raise ValueError("loss_funcs must be a dict or a callable")

        # Handle metric functions
        if isinstance(metric_funcs, dict):
            self.metric_funcs = metric_funcs
        elif callable(metric_funcs):
            self.metric_funcs = {"Metric": metric_funcs}
        else:
            self.metric_funcs = None

        # Handle loss weights
        if multi_loss_weights is not None:
            assert len(multi_loss_weights) == len(
                self.loss_funcs
            ), "Mismatch between number of loss functions and weights"
            self.multi_loss_weights = multi_loss_weights
        else:
            self.multi_loss_weights = [1.0 for _ in self.loss_funcs]

        # Handle regularizer
        if regularizer is not None:
            self.lambda_factor = regularizer.get("lambda", 0.0)
            assert (
                self.lambda_factor >= 0
            ), "Lambda factor for regularization must be non-negative"
        else:
            self.lambda_factor = 0.0

        # Optimizer, scheduler, and switcher
        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params
        self.scheduler_cls = scheduler_cls
        self.scheduler_params = scheduler_params
        self.switcher = switcher
        self.current_epoch_switch = switcher["epoch"] if switcher else None
        self.new_optimizer_cls = switcher["optimizer"] if switcher else None

        # Weights function
        self.weights_func = weights_func

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
        optimizer = self.optimizer_cls(
            self.parameters(), **self.hparams.optimizer_params
        )
        schedulers = []
        if self.scheduler_cls and self.scheduler_params:
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_params)
            schedulers.append({"scheduler": scheduler, "interval": "epoch"})
        return [optimizer], schedulers

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        loss_dict = {}
        total_loss = 0.0
        for i, (key, loss_func) in enumerate(self.loss_funcs.items()):
            loss = loss_func(outputs, targets) * self.multi_loss_weights[i]
            loss_dict[key] = loss
            total_loss += loss

        # L2 Regularization
        if self.lambda_factor > 0:
            l2_reg = self.lambda_factor * sum(
                p.pow(2.0).sum() for p in self.model.parameters()
            )
            loss_dict["L2_reg"] = l2_reg
            total_loss += l2_reg

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
            loss = loss_func(outputs, targets) * self.multi_loss_weights[i]
            loss_dict[key] = loss
            total_loss += loss

        # L2 Regularization (optional in validation)
        if self.lambda_factor > 0:
            l2_reg = self.lambda_factor * sum(
                p.pow(2.0).sum() for p in self.model.parameters()
            )
            loss_dict["L2_reg"] = l2_reg
            total_loss += l2_reg

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

    def on_epoch_end(self):
        # Handle optimizer switching if specified
        if self.switcher and self.current_epoch_switch == self.current_epoch:
            optimizer = self.optimizers()
            new_optimizer = self.new_optimizer_cls(self.model.parameters())
            self.trainer.replace_optimizer(new_optimizer)
            self.current_epoch_switch = None  # Prevent multiple switches
            self.log(
                "info",
                f"Optimizer switched to {self.new_optimizer_cls.__name__}",
                on_epoch=True,
            )

        # Update loss weights if weights_func is provided
        if self.weights_func:
            self.multi_loss_weights = self.weights_func(
                self.multi_loss_weights, self.current_epoch
            )
            self.log_hyperparams({"multi_loss_weights": self.multi_loss_weights})

    def on_train_epoch_end(self):
        # Example: Log average metrics over the epoch
        if self.metric_funcs:
            for key, metrics in self.train_metrics_history.items():
                avg_metric = torch.stack(metrics).mean()
                self.log(f"train/avg_{key}", avg_metric, prog_bar=True, logger=True)
            # Clear metrics history after logging
            self.train_metrics_history = {key: [] for key in self.metric_funcs}

    def on_validation_epoch_end(self):
        # Example: Log average metrics over the epoch
        if self.metric_funcs:
            for key, metrics in self.val_metrics_history.items():
                avg_metric = torch.stack(metrics).mean()
                self.log(f"val/avg_{key}", avg_metric, prog_bar=True, logger=True)
            # Clear metrics history after logging
            self.val_metrics_history = {key: [] for key in self.metric_funcs}
