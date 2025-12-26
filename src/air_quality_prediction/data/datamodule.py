from typing import Any, Dict, Optional

import lightning as L
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from air_quality_prediction.preprocessing.preprocessing import preprocess_raw_dataframe


class AQIDataModule(L.LightningDataModule):
    """
    LightningDataModule for air quality prediction.

    Handles:
        - Loading raw CSV
        - Preprocessing (imputation, encoding, etc.)
        - Train/val/test splitting
        - DataLoader construction
    """

    def __init__(
        self,
        csv_path: str,
        batch_size: int = None,
        test_size: float = None,
        val_size: float = None,
        target_col: str = None,
        random_state: int = 42,
        num_workers: int = 0,
    ):
        """
        Args:
            csv_path (str): Path to raw CSV file.
            batch_size (int): Batch size for DataLoaders. Defaults to 64.
            test_size (float): Proportion of data for test set. Defaults to 0.25.
            val_size (float): Proportion of *train* data for validation. Defaults to 0.25.
            target_col (str): Name of target column. Defaults to "european_aqi".
            random_state (int): Random seed for reproducibility. Defaults to 42.
            num_workers (int): Number of subprocesses for data loading. Defaults to 0.
        """
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.target_col = target_col
        self.random_state = random_state
        self.num_workers = num_workers

        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None
        self.metadata: Dict[str, Any] = {}

    def setup(self, stage: str):
        """
        Load and preprocess data. Called on every GPU in distributed training.
        """
        if stage in ("fit", "validate", "test", "predict"):
            df = pd.read_csv(self.csv_path)
            X, y, label_encoders = preprocess_raw_dataframe(df, target_col=self.target_col)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.val_size, random_state=self.random_state
            )

            X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
            y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
            X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
            y_val_t = torch.tensor(y_val.values, dtype=torch.float32)
            X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
            y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

            self.train_dataset = TensorDataset(X_train_t, y_train_t)
            self.val_dataset = TensorDataset(X_val_t, y_val_t)
            self.test_dataset = TensorDataset(X_test_t, y_test_t)

            self.metadata = {
                "input_size": X_train.shape[1],
                "label_encoders": label_encoders,
                "feature_names": list(X.columns),
                "n_train": len(self.train_dataset),
                "n_val": len(self.val_dataset),
                "n_test": len(self.test_dataset),
            }

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        assert self.train_dataset is not None, "setup() not called or train_dataset is None"
        return self._init_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        assert self.val_dataset is not None, "setup() not called or val_dataset is None"
        return self._init_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        assert self.test_dataset is not None, "setup() not called or test_dataset is None"
        return self._init_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """Same as test_dataloader for now (can be extended for inference-only data)."""
        return self.test_dataloader()

    def _init_dataloader(self, dataset: TensorDataset, shuffle: bool) -> DataLoader:
        """Helper to initialize a DataLoader (like `init_dataloader` in example)."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    @property
    def input_size(self) -> int:
        """Convenient access to input dimension."""
        return self.metadata.get("input_size", 0)

    @property
    def label_encoders(self) -> Dict[str, Any]:
        """Access fitted label encoders."""
        return self.metadata.get("label_encoders", {})
