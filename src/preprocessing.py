"""Data loading, splitting, and preprocessing utilities."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreparedData:
    """Raw and transformed versions of one reproducible train/test split."""

    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: list[str]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load an invoice CSV and fail clearly when it is missing."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    categorical_dtypes = {
        "invoice_id": "string",
        "company_code": "string",
        "customer_id": "string",
        "customer_group": "string",
        "region": "string",
        "industry": "string",
        "currency": "string",
        "invoice_channel": "string",
    }
    return pd.read_csv(dataset_path, dtype=categorical_dtypes)


def split_features_target(
    df: pd.DataFrame, target_column: str = "target_paid_late"
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, excluding row and high-cardinality identifiers."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing")
    columns_to_drop = [target_column]
    columns_to_drop.extend(
        column for column in ("invoice_id", "customer_id") if column in df.columns
    )
    return df.drop(columns=columns_to_drop), df[target_column].astype(int)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build one-hot categorical and scaled numeric transformations."""
    categorical_columns = [
        column
        for column in X.columns
        if is_string_dtype(X[column].dtype)
        or isinstance(X[column].dtype, pd.CategoricalDtype)
        or is_bool_dtype(X[column].dtype)
    ]
    numeric_columns = [
        column
        for column in X.columns
        if is_numeric_dtype(X[column].dtype) and not is_bool_dtype(X[column].dtype)
    ]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("onehot", encoder)])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_train_test_data(
    df: pd.DataFrame,
    target_column: str = "target_paid_late",
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    """Create a stratified split and fit preprocessing only on training data."""
    X, y = split_features_target(df, target_column)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    preprocessor = build_preprocessor(X_train_raw)
    X_train = np.asarray(preprocessor.fit_transform(X_train_raw), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=np.float32)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return PreparedData(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )
