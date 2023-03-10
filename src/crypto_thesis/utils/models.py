# -*- coding: utf-8 -*-
from typing import List, Tuple

import pandas as pd


def mt_split_train_test(master_table: pd.DataFrame,
                        index_col: str,
                        train_test_cutoff_date: str,
                        target_col: List[str]) -> \
                                        Tuple[pd.DataFrame,
                                            pd.DataFrame,
                                            pd.DataFrame,
                                            pd.DataFrame]:
    """Split master table into train and test datasets"""

    # model adjustment: labeling with 0 and 1
    master_table = master_table.replace({"top": 1, "bottom": 0})
    master_table = master_table.set_index(index_col)

    # split into train and test considering time series logic
    master_table_train = master_table[master_table["close_time"] < train_test_cutoff_date]
    master_table_test = master_table[master_table["close_time"] >= train_test_cutoff_date]

    # drop not useful columns
    master_table_train = master_table_train.drop(columns=["close_time"])
    master_table_test = master_table_test.drop(columns=["close_time"])

    # split into train and test
    X_train = master_table_train.drop(columns=target_col)
    y_train = master_table_train[target_col]
    X_test = master_table_test.drop(columns=target_col)
    y_test = master_table_test[target_col]

    return X_train, y_train, X_test, y_test
