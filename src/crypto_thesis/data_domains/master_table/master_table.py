# -*- coding: utf-8 -*-
import logging
from typing import Dict, Tuple

import pandas as pd
from imblearn.under_sampling import NearMiss

from crypto_thesis.utils import mt_split_train_test, scale_train_test

logger = logging.getLogger(__name__)
TARGET_COL = ["label"]
INDEX_COL = "window_nbr"


def build_master_table(fte_df: pd.DataFrame,
                        spine: pd.DataFrame,
                        class_bounds: Dict[str, float],
                        train_test_cutoff_date: str) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                              pd.DataFrame, pd.DataFrame]:
    """Builds master table (features and target)

    Args:
        fte_df (pd.DataFrame): dataframe with features
        spine (pd.DataFrame): dataframe with target
        class_bounds (Dict[str, float]): classes bounds value to determine if target is balanced or not
        topN_features (int): amount of features to select

    Returns:
        pd.DataFrame: dataframe representing the master table
    """

    # drop unnecessary column for this step
    spine = spine.drop(columns=["volume_cumsum"])

    master_table = fte_df.merge(spine, on=["open_time", "close_time"], how="inner")
    assert master_table.shape[0] == fte_df.shape[0] == spine.shape[0], \
            "Mismatch of dates between features and spine, review."

    master_table_dropped = master_table.dropna()

    master_table_numbered = _build_window_numbers(df=master_table_dropped)
    window_nbr_lookup = master_table_numbered[["window_nbr", "open_time", "close_time", "target_time"]]
    # drop useless columns (they're in spine layer if any troubleshooting needed)
    master_table_numbered = master_table_numbered.drop(columns=["target_time_log_return",
                                                                "std",
                                                                "logret_cumsum",
                                                                "target_time_close",
                                                                "close_time_close",
                                                                "close_to_tgt_time_logret",
                                                                "pctchg_cumsum",
                                                                "close_to_tgt_time_pctchg",
                                                                "open_time",
                                                                "target_time"])

    logger.info("Scaling features")
    X_train, y_train, X_test, y_test = mt_split_train_test(master_table=master_table_numbered,
                                                            index_col=INDEX_COL,
                                                            train_test_cutoff_date=train_test_cutoff_date,
                                                            target_col=TARGET_COL)

    logger.info("Checking for class unbalancing")
    train_df_bal = mt_balance_classes(X=X_train,
                                      y=y_train,
                                        class_bounds=class_bounds)
    X_train_bal, y_train_bal = train_df_bal.drop(columns=TARGET_COL), train_df_bal[TARGET_COL]

    X_train_bal, X_test = scale_train_test(X_train=X_train_bal, X_test=X_test)

    train_df_bal = X_train_bal.merge(y_train_bal, left_index=True, right_index=True, how="inner")
    train_df_bal = train_df_bal.reset_index()
    test_df = X_test.merge(y_test, left_index=True, right_index=True, how="inner")
    test_df = test_df.reset_index()

    # retrieve window_nbr after class balancing
    window_nbr_lookup_train = window_nbr_lookup[window_nbr_lookup["window_nbr"].isin(train_df_bal["window_nbr"])]
    window_nbr_lookup_test = window_nbr_lookup[window_nbr_lookup["window_nbr"].isin(test_df["window_nbr"])]

    logger.info("Checking master table quality")
    _check_master_table_quality(df=train_df_bal,
                                class_bounds=class_bounds)

    return train_df_bal, window_nbr_lookup_train, test_df, window_nbr_lookup_test


def build_master_table_oos(fte_df: pd.DataFrame,
                        spine: pd.DataFrame) -> pd.DataFrame:

    master_table = fte_df.merge(spine, on=["open_time", "close_time"], how="inner")
    assert master_table.shape[0] == fte_df.shape[0] == spine.shape[0], \
            "Mismatch of dates between features and spine, review."

    master_table_dropped = master_table.dropna()
    master_table_numbered = _build_window_numbers(df=master_table_dropped)
    window_nbr_lookup = master_table_numbered[["window_nbr", "open_time", "close_time", "target_time"]]

    # drop useless columns (they're in spine layer if any troubleshooting needed)
    master_table_numbered = master_table_numbered.drop(columns=["target_time_log_return",
                                                                "std",
                                                                "logret_cumsum",
                                                                "target_time_close",
                                                                "close_time_close",
                                                                "close_to_tgt_time_logret",
                                                                "pctchg_cumsum",
                                                                "close_to_tgt_time_pctchg",
                                                                "open_time",
                                                                "target_time",
                                                                "close_time",
                                                                "volume_cumsum"])

    master_table_numbered = master_table_numbered.set_index(INDEX_COL)
    X_train_bal, y_train_bal = master_table_numbered.drop(columns=TARGET_COL), master_table_numbered[TARGET_COL]
    X_test = X_train_bal.copy()

    X_train_bal, _ = scale_train_test(X_train=X_train_bal, X_test=X_test)

    mt = X_train_bal.merge(y_train_bal, left_index=True, right_index=True, how="inner")
    mt = mt.replace({"top": 1, "bottom": 0}).reset_index()

    return mt, window_nbr_lookup


def _build_window_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Build incremental value representing the index for each window number

    Args:
        df (pd.DataFrame): dataframe with defined start and end timestamps

    Returns:
        pd.DataFrame: dataframe with windows numbered
    """

    df = df.sort_values(by="close_time", ascending=True)
    df = df.reset_index(drop=True)
    idxs = [idx + 1 for idx in  df.index]

    df.loc[:, "window_nbr"] = idxs

    return df


def _check_master_table_quality(df: pd.DataFrame,
                                class_bounds: Dict[str, float]) -> None:
    """Quality checks in the master table

    Args:
        df (pd.DataFrame): dataframe representing the master table
        class_bounds (Dict[str, float]): classes bounds value to determine if target is balanced or not

    Returns:
        None. Raises error if criteria isn't met
    """

    # check label unbalancing
    labels_pct = (df.label.value_counts() / df.shape[0]).values
    assert any([(label_pct >= class_bounds["lower"] and \
                    label_pct <= class_bounds["upper"]) \
                for label_pct in labels_pct]), "Unbalanced classes, review."

    # check nulls
    assert df.isnull().sum().sum() == 0, "Master table contains null, review."


def mt_balance_classes(X: pd.DataFrame,
                       y: pd.DataFrame,
                        class_bounds: Dict[str, float]) -> pd.DataFrame:
    """Balance master table's classes

    Args:
        df (pd.DataFrame): dataframe with classes to be balanced
        class_bounds (Dict[str, float]): classes bounds value to determine if target is balanced or not
        topN_features (int): amount of features to select

    Returns:
        pd.DataFrame: dataframe representing balanced master table
    """

    logger.info("Checking for class balance")
    label0_count, _ = y.label.value_counts()
    label0_pct = label0_count / y.shape[0]

    # check if any label is outside the desired range
    # there's no need to check both labels, if 1 is outside the range, the other will also be
    if not pd.Series(label0_pct). \
            between(class_bounds["lower"],
                    class_bounds["upper"]) \
                    [0]: #pull index [0] always works because it's only 1 label

        df = _balance_classes(X=X, y=y)

    else:
        logger.info("Class are balanced, skipping balancing method")

    return df


def _balance_classes(X: pd.DataFrame,
                     y: pd.DataFrame) -> pd.DataFrame:
    """Balance target classes using NearMiss (kNN) method

    Args:
        X (pd.DataFrame): dataframe with features
        y (pd.DataFrame): dataframe with target

    Returns:
        pd.DataFrame: dataframe representing balanced master table
    """

    nm = NearMiss(version=3)
    X_res, y_res = nm.fit_resample(X, y)

    idxs = nm.sample_indices_ + 1 #sum 1 because it starts with 0
    X_res.index, y_res.index = idxs, idxs

    mt = X_res.merge(y_res, left_index=True, right_index=True, how="inner")
    mt.index.name = "window_nbr"

    return mt
