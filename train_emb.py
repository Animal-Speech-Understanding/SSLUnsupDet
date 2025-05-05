import argparse
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim
from torch.utils.data import DataLoader

from ssl_model.dataset import SpermWhaleClicksDataset
from ssl_model.losses import NoiseContrastiveEstimationLoss
from ssl_model.models import SpectralBoundaryEncoder
from ssl_model.trainer import SpectralBoundaryLightningModule
from ssl_model.utils import seed_everything


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
    train_set = SpermWhaleClicksDataset(
        n_samples = dataset_params["train_samples"],
        base_path = dataset_params["wavs_path"],
        subset = "train",
        window_sec = dataset_params["window"],
        pad_frames = dataset_params["window_pad"],
        sample_rate = dataset_params["sample_rate"],
        epsilon = dataset_params["epsilon"],
        seed = dataset_params["seed"],
    )

    val_set = SpermWhaleClicksDataset(
        n_samples=dataset_params["val_samples"],
        base_path=dataset_params["wavs_path"],
        subset="val",
        window_sec=dataset_params["window"],
        pad_frames=dataset_params["window_pad"],
        sample_rate=dataset_params["sample_rate"],
        epsilon=dataset_params["epsilon"],
        seed=dataset_params["seed"],
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=training_params["batch_size"],
        sampler=train_set.sampler,
        num_workers=4,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=training_params["batch_size"],
        num_workers=4,
        persistent_workers=True,
    )

    # Initialize model
    model = SpectralBoundaryEncoder(**model_params)

    optimizer_cls = optim.Adam

    # Initialize loss function
    n_negatives = training_params.get("n_negatives")
    nce_loss = NoiseContrastiveEstimationLoss(n_negatives=n_negatives)
    loss_funcs = {"NE Loss": nce_loss}

    # Initialize Lightning Module
    lightning_module = SpectralBoundaryLightningModule(
        model=model,
        optimizer_cls=optimizer_cls,
        loss_funcs=loss_funcs,
    )

    # Define Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_dir, "checkpoints"),
        filename="best-checkpoint",
        save_top_k=2,
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
