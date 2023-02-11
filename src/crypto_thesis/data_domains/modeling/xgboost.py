# -*- coding: utf-8 -*-

from datetime import datetime
from typing import Tuple

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TARGET_COL = ["label"]
# these cols were useful so far, but not anymore
INDEX_COL = "window_nbr"


def xgboost_model_fit(master_table: pd.DataFrame) -> Tuple[XGBClassifier,
                                                        pd.DataFrame, pd.DataFrame,
                                                        pd.DataFrame, pd.DataFrame]:

    # model adjustment: labeling with 0 and 1
    master_table = master_table.replace({"top": 1, "bottom": 0})
    master_table = master_table.set_index(INDEX_COL)

    X = master_table.drop(columns=TARGET_COL)
    y = master_table[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    model.fit(X_train, y_train)

    return model, X_train, y_train, X_test, y_test


def xgboost_model_predict(model: XGBClassifier, X_test: pd.DataFrame) -> pd.DataFrame:

    idxs = X_test.index.tolist()
    y_pred = model.predict(X_test)
    return pd.DataFrame(data={"y_pred": y_pred}, index=idxs)


def xgboost_model_reporting(model: XGBClassifier,
                            X_test: pd.DataFrame,
                            y_test: pd.DataFrame,
                            y_pred: pd.DataFrame,
                            model_data_interval: str) -> pd.DataFrame:

    # get model's accuracy
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)

    # get model's parameters
    params = model.get_xgb_params()

    # get model's probability
    idxs = X_test.index.tolist()
    probas = model.predict_proba(X_test)
    probas_df = pd.DataFrame(data=probas, index=idxs, columns=["proba_label_0", "proba_label_1"])

    reporting_df = pd.DataFrame({"runtime_brtz": str(datetime.now()),
                                "accuracy": acc,
                                "model_params": str(params),
                                "data_interval": model_data_interval,
                                "probas": str(probas_df.to_dict(orient="index"))
                                }, index=[0])

    return reporting_df
