# -*- coding: utf-8 -*-
from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

TARGET_COLS = ["label"]
# these cols were useful so far, but not anymore
USELESS_COLS = ["window_nbr"]

def xgboost_model(master_table: pd.DataFrame, bars_window_params: Dict[str, int]) -> pd.DataFrame:

    master_table = master_table.replace({"top": 1, "bottom": 0})

    lookbehind_window = bars_window_params["bars_accum_lookbehind"]
    lookahead_predict = bars_window_params["bars_predict_ahead"]

    X = master_table[master_table["window_nbr"] <= lookbehind_window]
    y = master_table[master_table["window_nbr"] == (lookbehind_window + lookahead_predict)]

    X_train = X.drop(columns=TARGET_COLS + USELESS_COLS)
    y_train = X[TARGET_COLS]

    X_test = y.drop(columns=TARGET_COLS + USELESS_COLS)
    y_test = y[TARGET_COLS]

    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
