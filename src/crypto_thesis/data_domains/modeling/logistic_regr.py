# -*- coding: utf-8 -*-

import math
import warnings
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from crypto_thesis.utils import mt_split_train_test

warnings.filterwarnings("ignore")

TARGET_COL = ["label"]
# these cols were useful so far, but not anymore
INDEX_COL = "window_nbr"


def logistic_regr_model_fit(master_table: pd.DataFrame,
                    train_test_cutoff_date: str,
                    # model_params: Dict[str, Any]
                    ) -> Tuple[LogisticRegression,
                                pd.DataFrame, pd.DataFrame,
                                pd.DataFrame, pd.DataFrame]:

    X_train, y_train, X_test, y_test = mt_split_train_test(master_table=master_table,
                                                            index_col=INDEX_COL,
                                                            train_test_cutoff_date=train_test_cutoff_date,
                                                            target_col=TARGET_COL)

    solvers = ["newton-cg", "lbfgs", "liblinear"]
    penalty = ["l2"]
    c_values = [100, 10, 1.0, 0.1, 0.01]

    grid = dict(solver=solvers, penalty=penalty, C=c_values)

    model = LogisticRegression()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X_train, y_train)

    model.set_params(**grid_result.best_params_)
    model.fit(X_train, y_train)

    return model, X_train, y_train, X_test, y_test


def logistic_regr_model_predict(model: LogisticRegression, X_test: pd.DataFrame) -> pd.DataFrame:

    idxs = X_test.index.tolist()
    y_pred = model.predict(X_test)
    return pd.DataFrame(data={"y_pred": y_pred}, index=idxs)


def logistic_regr_model_reporting(model: LogisticRegression,
                            X_test: pd.DataFrame,
                            y_test: pd.DataFrame,
                            y_pred: pd.DataFrame,
                            master_table: pd.DataFrame,
                            model_data_interval: str,
                            spine_preproc_params: Dict[str, Any],
                            spine_label_params: Dict[str, Any],
                            train_test_cutoff_date: str,
                            slct_topN_features: int,
                            min_years_existence: int) -> pd.DataFrame:

    # get model's accuracy
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)

    # get model's parameters
    params = model.get_params()

    # get model's probability
    idxs = X_test.index.tolist()
    probas = model.predict_proba(X_test)
    probas_df = pd.DataFrame(data=probas, index=idxs, columns=["proba_label_0", "proba_label_1"])

    # get features' importance
    weights = model.coef_[0]
    ftes = X_test.columns.tolist()
    fte_imps = pd.DataFrame({"ftes": ftes})
    fte_imps.loc[:, "importance"] = pow(math.e, weights)
    fte_imps = fte_imps.set_index("ftes").to_dict()

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
                                "label_class_balance": str(label_class_balance),
                                "topN_features_slct_qty": slct_topN_features,
                                "selected_tickers": str({"tickers": tickers}),
                                "min_historical_data_window_years": min_years_existence,
                                "confusion_matrix": repr(cm)
                                }, index=[0])

    return reporting_df
