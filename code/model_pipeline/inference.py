import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from joblib import load
from typing import Tuple, Union
from datetime import datetime
import xgboost as xgb
from preprocess import encode_and_update


ScalerType = Union[StandardScaler, MinMaxScaler]
MODEL_PATH = "../models/"
DATA_PATH = "../data/"


def load_joblibs() -> Tuple[pd.Index, OrdinalEncoder, ScalerType, xgb.XGBRegressor]:
    cols = load(MODEL_PATH + "columns.joblib")
    ordinal = load(MODEL_PATH + "Ordinal_Encoder.joblib")
    standard = load(MODEL_PATH + "Standard_Scaler.joblib")
    model = load(MODEL_PATH + "XGBoost_classifier.joblib")
    return cols, ordinal, standard, model


def make_predictions(data: pd.DataFrame) -> pd.DataFrame:
    time = datetime.today()
    cols, ordinal, standard, model = load_joblibs()
    ids = data["customerID"]
    df_test = data[cols]

    try:
        ordinal = encode_and_update(df_test, MODEL_PATH + "Ordinal_Encoder.joblib")
    except:
        ordinal = ordinal

    df_test[df_test.select_dtypes(include=["object"]).columns] = ordinal.transform(
        df_test.select_dtypes(include=["object"])
    )

    df_test = standard.transform(df_test)
    y_pred = model.predict(df_test)

    submission_df = pd.DataFrame({"customerID": ids, "Churn": y_pred})
    submission_df.to_csv(DATA_PATH + f"prediction_{time}.csv", index=False)
    print(f"Submission file created successfully.\nDateTime: {time}")
    return submission_df
