import argparse
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torch.utils.data import DataLoader

from dataset import SpermWhaleClicks
from losses import NoiseContrastiveEstimationLoss
from metrics import NoiseContrastiveEstimationMetric
from models import SpectralBoundaryEncoder
from PyFire import SpectralBoundaryLightningModule
from utils import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Train SpectralBoundaryEncoder with PyTorch Lightning")

    parser.add_argument("-c", "--config", type=str, required=True, help="JSON file for configuration")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    config_path = f"configs/{args.config}"
    seed = args.seed

    # Set random seed for reproducibility
    seed_everything(seed)

    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    save_dir = config["utils"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Dataset parameters
    dataset_params = config["dataset"]

    # Training parameters
    training_params = config["training"]

    # Model parameters
    model_params = config["model"]

    # Initialize datasets
    train_set = SpermWhaleClicks(
        n_samples=dataset_params["train_samples"],
        base_path=dataset_params["wavs_path"],
        subset="train",
        window=dataset_params["window"],
        window_pad=dataset_params["window_pad"],
        sample_rate=dataset_params["sample_rate"],
        epsilon=dataset_params["epsilon"],
        seed=dataset_params["seed"],
    )

    val_set = SpermWhaleClicks(
        n_samples=dataset_params["val_samples"],
        base_path=dataset_params["wavs_path"],
        subset="val",
        window=dataset_params["window"],
        window_pad=dataset_params["window_pad"],
        sample_rate=dataset_params["sample_rate"],
        epsilon=dataset_params["epsilon"],
        seed=dataset_params["seed"],
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=training_params["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=training_params["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Initialize model
    model = SpectralBoundaryEncoder(**model_params)

    # Define optimizer
    optimizer_config = training_params["optimizer"]
    optimizer_name = optimizer_config["name"].lower()

    if optimizer_name == "sgd":
        optimizer_cls = optim.SGD
        optimizer_params = {
            "lr": optimizer_config["learning_rate"],
            "momentum": optimizer_config.get("momentum", 0.0),
            "weight_decay": optimizer_config.get("weight_decay", 0.0),
        }
    elif optimizer_name == "adam":
        optimizer_cls = optim.Adam
        optimizer_params = {
            # "lr": optimizer_config["learning_rate"],
            # "weight_decay": optimizer_config.get("weight_decay", 0.0),
        }
    else:
        raise ValueError(f"Unsupported optimizer name: {optimizer_config['name']}")

    # Define scheduler
    scheduler_config = training_params.get("scheduler", None)
    if scheduler_config is not None:
        scheduler_type = scheduler_config["type"].lower()
        scheduler_kwargs = scheduler_config.get("kwargs", {})
        if scheduler_type == "plateau":
            scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
        elif scheduler_type == "step":
            scheduler_cls = torch.optim.lr_scheduler.StepLR
        elif scheduler_type == "multi_step":
            scheduler_cls = torch.optim.lr_scheduler.MultiStepLR
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
        scheduler_params = scheduler_kwargs
    else:
        scheduler_cls = None
        scheduler_params = None

    # Initialize loss function
    n_negatives = training_params.get("n_negatives", 2)
    print(f"n_negatives: {n_negatives}")

    nce_loss = NoiseContrastiveEstimationLoss(n_negatives=n_negatives)

    def loss_fx(z, targets):
        preds = nce_loss.compute_preds(z)
        loss = nce_loss.loss(preds)
        return loss

    # Initialize metric function
    nce_metric = NoiseContrastiveEstimationMetric()

    def metric_fx(z, targets):
        preds = nce_metric.compute_preds(z)
        metric = nce_metric.metric(preds)
        return metric

    loss_funcs = {"NE Loss": loss_fx}
    metric_funcs = {"NE Metric": metric_fx}

    # Initialize Lightning Module
    lightning_module = SpectralBoundaryLightningModule(
        model=model,
        optimizer_cls=optimizer_cls,
        optimizer_params=optimizer_params,
        scheduler_cls=scheduler_cls,
        scheduler_params=scheduler_params,
        loss_funcs=loss_funcs,
        metric_funcs=metric_funcs,
    )

    # Define Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, "checkpoints"),
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val/total_loss",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/total_loss", patience=10, verbose=True, mode="min"
    )

    # Initialize Logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(save_dir, "logs"), name="tensorboard"
    )

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=training_params["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        deterministic=True,
        precision=32,  # or 16 if using mixed precision
    )

    # Train the model
    trainer.fit(lightning_module, train_loader, val_loader)

    # Optionally, save the final model
    final_model_path = os.path.join(save_dir, "models", "final_model.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(lightning_module.model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
