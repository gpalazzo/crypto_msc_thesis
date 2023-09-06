# -*- coding: utf-8 -*-

import logging
import math
import warnings
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from crypto_thesis.utils import optimize_params

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

TARGET_COL = ["label"]
# these cols were useful so far, but not anymore
INDEX_COL = "window_nbr"


def logreg_model_fit(master_table_train: pd.DataFrame,
                    model_params: Dict[str, Any],
                    logreg_optimize_params: bool,
                    logreg_default_params: Dict[str, Any]
                    ) -> Tuple[LogisticRegression, pd.DataFrame]:
    """_summary_

    Args:
        master_table (pd.DataFrame): dataframe representing the master table
        train_test_cutoff_date (str): cutoff date for train/test split
        model_params (Dict[str, Any]): current model parameters grid
        logreg_optimize_params (bool): whether if it's to optimize or not parameters
        logreg_default_params (Dict[str, Any]): logreg model default parameters

    Returns:
        Tuple[LogisticRegression, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: trained model object,
        best model parameters, features train, target train, features test and target test, respectively
    """

    X_train, y_train = master_table_train.drop(columns=TARGET_COL), master_table_train[TARGET_COL]

    model = LogisticRegression(**logreg_default_params)

    if logreg_optimize_params:
        # params opt
        logger.info("Optimzing parameters")
        params_opt = optimize_params(model=model,
                                    grid=model_params,
                                    X_train=X_train,
                                    y_train=y_train,
                                    n_splits=10,
                                    n_repeats=3)
        params_opt = params_opt.best_params_

    else:
        params_opt = model_params.copy()

    params_opt.update(logreg_default_params)
    model.set_params(**params_opt)
    model.fit(X_train, y_train)

    df_params_opt = pd.DataFrame(params_opt, index=[0])

    return model, df_params_opt


def logreg_model_predict(model: LogisticRegression, master_table_test: pd.DataFrame) -> pd.DataFrame:
    """LogReg model prediction

    Args:
        model (LogisticRegression): LogReg trained classifier
        X_test (pd.DataFrame): dataframe with features test

    Returns:
        pd.DataFrame: dataframe with model's prediction
    """

    X_test = master_table_test.drop(columns=TARGET_COL)

    idxs = X_test.index.tolist()
    y_pred = model.predict(X_test)

    return pd.DataFrame(data={"y_pred": y_pred}, index=idxs)


def logreg_model_reporting(model: LogisticRegression,
                            master_table_test: pd.DataFrame,
                            y_pred: pd.DataFrame,
                            model_data_interval: str,
                            spine_preproc_params: Dict[str, Any],
                            spine_label_params: Dict[str, Any],
                            train_test_cutoff_date: str,
                            slct_topN_features: int,
                            min_years_existence: int) -> pd.DataFrame:
    """LogReg model reporting

    Args:
        model (LogisticRegression): LogReg trained classifier
        X_test (pd.DataFrame): dataframe with features test
        y_test (pd.DataFrame): dataframe with target test
        y_pred (pd.DataFrame): dataframe with model's prediction
        master_table (pd.DataFrame): dataframe representing the master table
        model_data_interval (str): interval which the raw data was collected
        spine_preproc_params (Dict[str, Any]): parameters for spine pre-processing
        spine_label_params (Dict[str, Any]): parameters for labeling the target
        train_test_cutoff_date (str): date to cutoff datasets into train and test
        slct_topN_features (int): amount of features to keep
        min_years_existence (int): minimum time, in years, for ticker existence

    Returns:
        pd.DataFrame: dataframe with model's metrics
    """

    X_test, y_test = master_table_test.drop(columns=TARGET_COL), master_table_test[TARGET_COL]

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
    label_class_balance = (y_test["label"].value_counts() / y_test.shape[0]).to_dict()

    # get selected tickers
    tickers = []
    for col in X_test.columns.tolist():
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
