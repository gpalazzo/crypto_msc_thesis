# -*- coding: utf-8 -*-
import logging
from typing import Dict, Tuple

import pandas as pd
from imblearn.under_sampling import NearMiss

logger = logging.getLogger(__name__)


def build_master_table(fte_df: pd.DataFrame,
                        spine: pd.DataFrame,
                        class_bounds: Dict[str, float],
                        topN_features: int) -> pd.DataFrame:
    """Builds master table (features and target)

    Args:
        fte_df (pd.DataFrame): dataframe with features
        spine (pd.DataFrame): dataframe with target
        class_bounds (Dict[str, float]): classes bounds value to determine if target is balanced or not
        topN_features (int): amount of features to select

    Returns:
        pd.DataFrame: dataframe representing the master table
    """

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

    logger.info("Checking for class unbalancing")
    master_table_numbered = mt_balance_classes(df=master_table_numbered,
                                               class_bounds=class_bounds,
                                               topN_features=topN_features)

    # retrieve window_nbr after class balancing
    window_nbr_lookup = window_nbr_lookup[window_nbr_lookup["window_nbr"].isin(master_table_numbered["window_nbr"])]

    logger.info("Checking master table quality")
    _check_master_table_quality(df=master_table_numbered,
                                class_bounds=class_bounds)

    return master_table_numbered, window_nbr_lookup


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


def mt_balance_classes(df: pd.DataFrame,
                        class_bounds: Dict[str, float],
                        topN_features: int) -> pd.DataFrame:
    """Balance master table's classes

    Args:
        df (pd.DataFrame): dataframe with classes to be balanced
        class_bounds (Dict[str, float]): classes bounds value to determine if target is balanced or not
        topN_features (int): amount of features to select

    Returns:
        pd.DataFrame: dataframe representing balanced master table
    """

    logger.info("Checking for class balance")
    label0_count, _ = df.label.value_counts()
    label0_pct = label0_count / df.shape[0]

    # check if any label is outside the desired range
    # there's no need to check both labels, if 1 is outside the range, the other will also be
    if not pd.Series(label0_pct). \
            between(class_bounds["lower"],
                    class_bounds["upper"]) \
                    [0]: #pull index [0] always works because it's only 1 label

        X, y, window_nbr_closetime = _split_master_table(df=df,
                                                        topN_features=topN_features)
        df = _balance_classes(X=X, y=y)

        df = df.merge(window_nbr_closetime, left_index=True, right_index=True, how="inner") \
                .reset_index() \
                .rename(columns={"index": "window_nbr"})

    else:
        logger.info("Class are balanced, skipping balancing method")

    return df


def _balance_classes(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
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

    return mt


def _split_master_table(df: pd.DataFrame, topN_features: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split master table into train, test and window_nbr

    Args:
        df (pd.DataFrame): dataframe representing the master table
        topN_features (int): amount of features to select

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: dataframe with features, target and window_nbr, respectively
    """

    y = df[["window_nbr", "label"]].set_index("window_nbr")
    X = df.drop(columns=["close_time", "label"]).set_index("window_nbr")

    # shape of X after split must be the same as the amount of selected features
    assert X.shape[1] == topN_features, "Wrong number of features in master table split, review"

    return X, y, df[["window_nbr", "close_time"]].set_index("window_nbr")
