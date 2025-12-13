from pathlib import Path
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from air_quality_prediction.preprocessing.preprocessing import preprocess_raw_dataframe
import logging

logger = logging.getLogger(__name__)

def create_data_loaders(
    csv_path: Path,
    test_size: float = 0.25,
    val_size: float = 0.25,
    random_state: int = 42,
    batch_size: int = 64,
    target_col: str = "european_aqi",
    id_col: str = "city_id"
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    End-to-end pipeline: load CSV → preprocess → split → DataLoader.
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Preprocess
    X, y, label_encoders = preprocess_raw_dataframe(
        df, target_col=target_col, id_col=id_col
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

    # Create datasets & loaders
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    metadata = {
        "input_size": X_train.shape[1],
        "label_encoders": label_encoders,
        "feature_names": list(X.columns),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }

    logger.info(
        f"✅ DataLoaders created: "
        f"train={metadata['n_train']}, "
        f"val={metadata['n_val']}, "
        f"test={metadata['n_test']}, "
        f"input_size={metadata['input_size']}"
    )

    return train_loader, val_loader, test_loader, metadata
