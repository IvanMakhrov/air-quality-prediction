import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from air_quality_prediction.models.base_model import NeuralNet

logger = logging.getLogger(__name__)


class AQILightningModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = NeuralNet(
            input_size=input_size, hidden_size=hidden_size, dropout_rate=dropout_rate
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_size: int,
    max_epochs: int = 50,
    hidden_size: int = 128,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    checkpoint_dir: Path = Path("models"),
    seed: int = 42,
) -> AQILightningModel:
    """
    Trains AQILightningModel and saves best checkpoint.

    Args:
        train_loader, val_loader: PyTorch DataLoaders
        input_size: number of input features
        ... other hyperparameters
        checkpoint_dir: where to save best model (DVC-tracked)

    Returns:
        Trained AQILightningModel
    """
    pl.seed_everything(seed, workers=True)
    checkpoint_dir.mkdir(exist_ok=True)

    model = AQILightningModel(
        input_size=input_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
    )

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-aqi-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        save_top_k=1,
        mode="min",
    )

    early_stop = pl.callbacks.EarlyStopping(monitor="val/loss", patience=5, mode="min")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop],
        accelerator="auto",
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    logger.info(f"🚀 Starting training (input_size={input_size}, hidden_size={hidden_size})")
    trainer.fit(model, train_loader, val_loader)

    logger.info(f"🏁 Training finished. Best model: {checkpoint_callback.best_model_path}")
    return model
