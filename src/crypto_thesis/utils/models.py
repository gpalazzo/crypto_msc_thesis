# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from xgboost import XGBClassifier


def mt_split_train_test(master_table: pd.DataFrame,
                        index_col: str,
                        train_test_cutoff_date: str,
                        target_col: List[str]) -> Tuple[pd.DataFrame,
                                            pd.DataFrame,
                                            pd.DataFrame,
                                            pd.DataFrame]:
    """Split master table into train and test datasets

    Args:
        master_table (pd.DataFrame): dataframe representing the master table
        index_col (str): column name to index the dataframe
        train_test_cutoff_date (str): cutoff date to split dataset into train and test
        target_col (List[str]): column with the target value

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: dataframes with features and target for training,
        and features and target for testing, respectively
    """

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


def optimize_params(model: Union[LogisticRegression, XGBClassifier],
                    grid: Dict[str, Any],
                    X_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    n_splits: int,
                    n_repeats: int,
                    random_state: int = 1,
                    grid_search_scoring: str = "accuracy") -> Dict[str, str]:
    """Optimize parameters using GridSearchCV method

    Args:
        model (Union[LogisticRegression, XGBClassifier]): model object
        grid (Dict[str, Any]): grid with parameters to be tested
        X_train (pd.DataFrame): dataframe with train features
        y_train (pd.DataFrame): dataframe with train target
        n_splits (int): number of splits for K-fold
        n_repeats (int): number of repeats for K-fold
        random_state (int, optional): random state for K-fold. Defaults to 1.
        grid_search_scoring (str, optional): method for scoring the grid search results. Defaults to "accuracy".

    Returns:
        Dict[str, str]: grid with chosen parameters
    """

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    grid_search = GridSearchCV(estimator=model,
                                param_grid=grid,
                                n_jobs=-1,
                                cv=cv,
                                scoring=grid_search_scoring,
                                error_score=0)

    grid_result = grid_search.fit(X_train, y_train)

    return grid_result
