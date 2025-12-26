import logging

import lightning as L
import torch
from torch import nn
from torchmetrics import MeanSquaredError, R2Score

logger = logging.getLogger(__name__)


class AQILightningModule(L.LightningModule):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)

        if stage == "train":
            self.train_mse(y_pred, y)
            self.train_r2(y_pred, y)
        elif stage == "val":
            self.val_mse(y_pred, y)
            self.val_r2(y_pred, y)
        elif stage == "test":
            self.test_mse(y_pred, y)
            self.test_r2(y_pred, y)

        return loss, y_pred

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, "train")
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_training_epoch_end(self):
        train_mse = self.train_mse.compute()
        train_rmse = torch.sqrt(train_mse)
        train_r2 = self.train_r2.compute()

        self.log("train/mse", train_mse, sync_dist=True)
        self.log("train/rmse", train_rmse, sync_dist=True)
        self.log("train/r2", train_r2, sync_dist=True)

        self.train_mse.reset()
        self.train_r2.reset()

    def validation_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, "val")
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        val_mse = self.val_mse.compute()
        val_rmse = torch.sqrt(val_mse)
        val_r2 = self.val_r2.compute()

        self.log("val/mse", val_mse, sync_dist=True)
        self.log("val/rmse", val_rmse, sync_dist=True)
        self.log("val/r2", val_r2, sync_dist=True)

        self.val_mse.reset()
        self.val_r2.reset()

    def test_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, "test")
        self.log("test/loss", loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        test_mse = self.test_mse.compute()
        test_rmse = torch.sqrt(test_mse)
        test_r2 = self.test_r2.compute()

        self.log("test/mse", test_mse)
        self.log("test/rmse", test_rmse)
        self.log("test/r2", test_r2)

        self.test_mse.reset()
        self.test_r2.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
