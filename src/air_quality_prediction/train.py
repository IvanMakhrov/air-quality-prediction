import git

# import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from air_quality_prediction.data.datamodule import AQIDataModule
from air_quality_prediction.models.base_model import BaseModel
from air_quality_prediction.models.module import AQILightningModule
from air_quality_prediction.models.tab_transformer import TabTransformer


def get_git_commit() -> str:
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:7]
    except Exception:
        return "unknown"


# @hydra.main(version_base=None, config_path="../../conf", config_name="config")
def train(cfg: DictConfig):
    commit_id = get_git_commit()

    datamodule = AQIDataModule(
        csv_path=cfg.data.csv_path,
        batch_size=cfg.data.batch_size,
        test_size=cfg.data.test_size,
        val_size=cfg.data.val_size,
        target_col=cfg.data.target_col,
        random_state=cfg.data.random_state,
        num_workers=cfg.data.num_workers,
    )

    datamodule.setup(stage="fit")

    L.seed_everything(cfg.model.seed, workers=True)

    if cfg.model.type == "base_model":
        model = BaseModel(input_size=datamodule.input_size, hidden_size=cfg.model.hidden_size)
    elif cfg.model.type == "tabtransformer":
        model = TabTransformer(
            n_num_features=datamodule.input_size,
            d_token=cfg.model.d_token,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    module = AQILightningModule(model)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name, tracking_uri=cfg.logging.tracking_uri
    )
    mlf_logger.log_hyperparams({"git_commit": commit_id})
    trainer = L.Trainer(max_epochs=cfg.model.max_epochs, logger=mlf_logger)
    trainer.fit(module, datamodule=datamodule)
    torch.save(module.model.state_dict(), cfg.output_file)


if __name__ == "__main__":
    train()
