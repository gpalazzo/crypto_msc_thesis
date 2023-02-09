# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

TARGET_COLS = ["target_time", "label"]
# these cols were useful so far, but not anymore
USELESS_COLS = ["open_time", "close_time"]

def xgboost_model(master_table: pd.DataFrame) -> pd.DataFrame:

    # breakpoint()

    for window_nbr in master_table["window_nbr"]:
        df_filter = master_table[master_table["close_time"].between(start, end)]

        X = df_filter.drop(columns=TARGET_COLS + USELESS_COLS)
        y = df_filter[TARGET_COLS]

        # breakpoint()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        # breakpoint()

        model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
        model.fit(X_train, y_train)

        # breakpoint()
