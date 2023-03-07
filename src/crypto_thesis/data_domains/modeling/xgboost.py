# -*- coding: utf-8 -*-

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

TARGET_COL = ["label"]
# these cols were useful so far, but not anymore
INDEX_COL = "window_nbr"


def xgboost_model_fit(master_table: pd.DataFrame,
                    train_test_cutoff_date: str,
                    model_params: Dict[str, Any]) -> Tuple[XGBClassifier,
                                                        pd.DataFrame, pd.DataFrame,
                                                        pd.DataFrame, pd.DataFrame]:

    # model adjustment: labeling with 0 and 1
    master_table = master_table.replace({"top": 1, "bottom": 0})
    master_table = master_table.set_index(INDEX_COL)

    # split into train and test considering time series logic
    master_table_train = master_table[master_table["close_time"] < train_test_cutoff_date]
    master_table_test = master_table[master_table["close_time"] >= train_test_cutoff_date]

    # drop not useful columns
    master_table_train = master_table_train.drop(columns=["close_time"])
    master_table_test = master_table_test.drop(columns=["close_time"])

    # split into train and test
    X_train = master_table_train.drop(columns=TARGET_COL)
    y_train = master_table_train[TARGET_COL]
    X_test = master_table_test.drop(columns=TARGET_COL)
    y_test = master_table_test[TARGET_COL]

    model = XGBClassifier(**model_params)
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
                            master_table: pd.DataFrame,
                            model_data_interval: str,
                            spine_preproc_params: Dict[str, Any],
                            spine_label_params: Dict[str, Any],
                            train_test_cutoff_date: str,
                            model_params: Dict[str, Any],
                            slct_topN_features: int,
                            min_years_existence: int) -> pd.DataFrame:

    # get model's accuracy
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)

    # get model's parameters
    params = model.get_xgb_params()

    # get model's probability
    idxs = X_test.index.tolist()
    probas = model.predict_proba(X_test)
    probas_df = pd.DataFrame(data=probas, index=idxs, columns=["proba_label_0", "proba_label_1"])

    # get features' importance
    fte_imps = model.get_booster().get_score(importance_type="weight")

    # get label class balance
    label_class_balance = (master_table["label"].value_counts() / master_table.shape[0]).to_dict()

    # get selected tickers
    tickers = []
    for col in master_table.columns.tolist():
        splitted = col.split("__")
        try:
            tickers.append(splitted[1])
        except IndexError:
            pass
    tickers = list(set(tickers))

    # confusion matrix
    # normalize = all means over rows and columns
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred, normalize="all")

    reporting_df = pd.DataFrame({"model_accuracy": acc,
                                "model_params": str(params),
                                "data_interval_collect": model_data_interval,
                                "test_probas": str(probas_df.to_dict(orient="index")),
                                "fte_importance": str(fte_imps),
                                "target_name": spine_preproc_params["target_name"],
                                "volume_bar_size": spine_preproc_params["volume_bar_size"],
                                "bar_ahead_predict": spine_preproc_params["bar_ahead_predict"],
                                "labeling_tau": spine_label_params["tau"],
                                "train_test_cutoff_date": train_test_cutoff_date,
                                "model_params": str(model_params),
                                "label_class_balance": str(label_class_balance),
                                "topN_features_slct_qty": slct_topN_features,
                                "selected_tickers": str({"tickers": tickers}),
                                "min_historical_data_window_years": min_years_existence,
                                "confusion_matrix": repr(cm)
                                }, index=[0])

    return reporting_df
