import logging

import lightning as L
import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MeanSquaredError, R2Score

from air_quality_prediction.data.datamodule import AQIDataModule
from air_quality_prediction.models.base_model import BaseModel
from air_quality_prediction.models.tab_transformer import TabTransformer

logger = logging.getLogger(__name__)


class AQILightningModule(L.LightningModule):
    def __init__(self, model_cfg: DictConfig, datamodule: AQIDataModule):
        super().__init__()
        self.model_cfg = model_cfg
        self.datamodule = datamodule
        self.model = None

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

    def configure_model(self):
        if self.model is not None:
            return
        input_size = self.datamodule.input_size

        if self.model_cfg.type == "base_model":
            self.model = BaseModel(
                input_size=input_size,
                hidden_size=self.model_cfg.hidden_size,
            )
        elif self.model_cfg.type == "tabtransformer":
            self.model = TabTransformer(
                n_num_features=input_size,
                d_token=self.model_cfg.d_token,
                n_layers=self.model_cfg.n_layers,
                n_heads=self.model_cfg.n_heads,
                dropout=self.model_cfg.dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_cfg.type}")

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
        self.log("train/r2", self.train_r2, on_epoch=True, sync_dist=True)

        if self.trainer.is_last_batch:
            self.log(
                "train/rmse", torch.sqrt(self.train_mse.compute()), on_epoch=True, sync_dist=True
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, "val")
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/r2", self.val_r2, on_epoch=True, sync_dist=True)
        if self.trainer.is_last_batch:
            self.log("val/rmse", torch.sqrt(self.val_mse.compute()), on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, "test")
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/r2", self.test_r2, on_epoch=True)
        if self.trainer.is_last_batch:
            self.log(
                "test/rmse", torch.sqrt(self.test_mse.compute()), on_epoch=True, sync_dist=True
            )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
