import subprocess
import sys

import fire
from hydra import compose, initialize

from air_quality_prediction.train import train


def download():
    """
    Download raw data using DVC: runs `dvc repro download`.
    Ensures DVC is installed and the `download` stage is defined in dvc.yaml.
    """

    try:
        subprocess.run([sys.executable, "-m", "dvc", "repro", "download"], check=True)
    except Exception:
        raise

    try:
        subprocess.run([sys.executable, "-m", "dvc", "push"], check=True)
    except Exception:
        raise


def train_model(overrides: list = None, config_path: str = "conf", config_name: str = "config"):
    """
    Запуск обучения модели.

    Args:
        overrides: список переопределений конфига (например: ["model=tabtransformer",
        "model.max_epochs=5"])
        config_path: путь к папке с конфигами (относительно корня проекта)
        config_name: имя основного конфига (без .yaml)
    """
    overrides = overrides or []

    with initialize(version_base=None, config_path="../../conf", job_name="train_job"):
        cfg = compose(config_name=config_name, overrides=overrides)

    train(cfg)


def main():
    fire.Fire(
        {
            "download": download,
            "train": lambda *args: train_model(list(args)),
        }
    )


if __name__ == "__main__":
    main()
