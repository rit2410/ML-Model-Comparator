# File: src/preprocess.py
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    class_names: List[str]

def _split_features_target(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def _build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), numeric_cols),
            ("cat", Pipeline(steps=cat_steps), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor

def prepare_dataset(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = True,
) -> PreparedData:
    """Prepare dataset: split, encode, scale; returns fitted preprocessor and splits."""
    X, y = _split_features_target(df, target_col)

    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        class_names = le.classes_.tolist()
    else:
        vals = sorted(pd.unique(y.dropna()))
        if len(vals) <= 20 and set(pd.Series(vals).dropna().astype(int)) == set(vals):
            class_names = [str(v) for v in vals]
            y_enc = y.astype(int).to_numpy()
        else:
            raise ValueError("Only classification tasks supported (categorical/discrete targets).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    preprocessor = _build_preprocessor(X_train, scale_numeric=scale_numeric)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_t, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_t, columns=feature_names)

    return PreparedData(
        X_train=X_train_df,
        X_test=X_test_df,
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        preprocessor=preprocessor,
        class_names=class_names,
    )
