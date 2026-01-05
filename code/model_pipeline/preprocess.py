import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from joblib import dump, load
from typing import Tuple, Union


MODEL_PATH = "../models/"
ScalerType = Union[StandardScaler, MinMaxScaler]


def read_prepare_df(
    PATH: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PATH)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
    df = df.dropna()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def ordinal_encoding(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ordinal = OrdinalEncoder()
    categorical_columns = X_train.select_dtypes(include=["object"]).columns
    ordinal.fit(X_train[categorical_columns])
    X_train[categorical_columns] = ordinal.transform(X_train[categorical_columns])
    X_test[categorical_columns] = ordinal.transform(X_test[categorical_columns])
    dump(ordinal, MODEL_PATH + "Ordinal_Encoder.joblib")
    dump(X_train.columns, MODEL_PATH + "columns.joblib")
    return X_train, X_test


def standardizing(
    X_train: pd.DataFrame, X_test: pd.DataFrame, scaler: ScalerType
) -> Tuple[np.ndarray, np.ndarray]:
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    dump(scaler, MODEL_PATH + "Standard_Scaler.joblib")
    return X_train, X_test


def encode_and_update(data: pd.DataFrame, ordinal_path: str) -> OrdinalEncoder:
    ordinal = load(ordinal_path)
    categorical_columns = data.select_dtypes(include=["object"]).columns
    for index, col in enumerate(categorical_columns):
        unique_items = set(data[col])
        known_items = set(ordinal.categories_[index])

        new_items = unique_items - known_items
        if new_items:
            ordinal.categories_[index] = np.append(
                ordinal.categories_[index], list(new_items)
            )

    dump(ordinal, MODEL_PATH + "Ordinal_Encoder.joblib")
    return ordinal
