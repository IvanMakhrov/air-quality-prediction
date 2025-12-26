import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def preprocess_raw_dataframe(
    df: pd.DataFrame, target_col: str = None
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
    """
    Applies full preprocessing pipeline to raw DataFrame.

    Steps:
        - Drop rows without target
        - Impute snow_depth with 0
        - Impute wind_speed_10m with regional-monthly median
        - Impute income & nitrogen_monoxide with regional-yearly median
        - Encode categorical features
        - Remove ID/target columns from features

    Returns:
        X: processed feature DataFrame (numeric)
        y: target Series
        label_encoders: dict of fitted LabelEncoders for categorical cols
    """
    logger.info("Starting preprocessing pipeline...")

    # 1. Ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # 2. Drop rows with missing target
    initial_len = len(df)
    df = df.dropna(subset=[target_col]).copy()
    logger.info(f"Dropped {initial_len - len(df)} rows with missing '{target_col}'")

    # 3. Snow depth: fill with 0
    if "snow_depth" in df.columns:
        n_null = df["snow_depth"].isna().sum()
        df["snow_depth"] = df["snow_depth"].fillna(0)
        if n_null > 0:
            logger.debug(f"Filled {n_null} NaNs in 'snow_depth' with 0")

    # 4. Wind speed: regional-monthly median
    if "wind_speed_10m" in df.columns:
        wind_median = df.groupby(["region", "year", "month"])["wind_speed_10m"].transform("median")
        n_null_before = df["wind_speed_10m"].isna().sum()
        df["wind_speed_10m"] = df["wind_speed_10m"].fillna(wind_median)
        n_null_after = df["wind_speed_10m"].isna().sum()
        if n_null_after > 0:
            # Fallback: global median
            global_median = df["wind_speed_10m"].median()
            df["wind_speed_10m"] = df["wind_speed_10m"].fillna(global_median)
            logger.warning(
                f"{n_null_after} wind_speed_10m values still NaN after group median — "
                f"filled with global median ({global_median:.2f})"
            )
        logger.debug(
            f"Filled {n_null_before - n_null_after} wind_speed_10m values with group median"
        )

    # 5. Income: regional-yearly median
    if "income" in df.columns:
        median_income = df.groupby(["region", "year"])["income"].transform("median")
        n_filled = df["income"].isna().sum()
        df["income"] = df["income"].fillna(median_income)
        logger.debug(f"Filled {n_filled} NaNs in 'income' with regional-yearly median")

    # 6. Nitrogen monoxide: regional-yearly median
    if "nitrogen_monoxide" in df.columns:
        median_no = df.groupby(["region", "year"])["nitrogen_monoxide"].transform("median")
        n_filled = df["nitrogen_monoxide"].isna().sum()
        df["nitrogen_monoxide"] = df["nitrogen_monoxide"].fillna(median_no)
        logger.debug(f"Filled {n_filled} NaNs in 'nitrogen_monoxide' with regional-yearly median")

    # 7. Prepare X, y
    y = df[target_col]
    cols_to_drop = [target_col]
    X = df.drop(columns=cols_to_drop)

    # 8. Encode categorical features
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        logger.debug(f"Encoded categorical column '{col}' (n_classes={len(le.classes_)})")

    logger.info(f"✅ Preprocessing complete. Final shape: X={X.shape}, y={y.shape}")
    return X, y, label_encoders
