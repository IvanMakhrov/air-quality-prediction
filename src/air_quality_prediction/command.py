import subprocess
import sys
from pathlib import Path

import fire
from hydra import compose, initialize

from air_quality_prediction.train import train


def ensure_data():
    data_path = Path("data/raw/air_weather_data_lite.csv")
    if not data_path.exists():
        print("Данные отсутствуют. Выполняется загрузка через DVC")
        result = subprocess.run(["dvc", "repro", "download"], check=True)
        if result.returncode != 0:
            print("Ошибка при выполнении `dvc repro download`:", result.stderr, file=sys.stderr)
            sys.exit(1)
        if not data_path.exists():
            raise RuntimeError(f"Данные всё ещё отсутствуют после загрузки: {data_path}")
    print("Данные готовы.")


def train_model(overrides: list = None, config_path: str = "conf", config_name: str = "config"):
    """
    Запуск обучения модели.

    Args:
        overrides: список переопределений конфига (например: ["model=tabpfn", "model.max_epochs=5"])
        config_path: путь к папке с конфигами (относительно корня проекта)
        config_name: имя основного конфига (без .yaml)
    """
    overrides = overrides or []

    ensure_data()

    with initialize(version_base=None, config_path="../../conf", job_name="train_job"):
        cfg = compose(config_name=config_name, overrides=overrides)

    train(cfg)


def main():
    fire.Fire(
        {
            "train": lambda *args: train_model(list(args)),
        }
    )


if __name__ == "__main__":
    main()
