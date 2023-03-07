# -*- coding: utf-8 -*-
from typing import Dict

import pandas as pd


def build_master_table(fte_df: pd.DataFrame,
                        spine: pd.DataFrame,
                        class_bounds: Dict[str, float]) -> pd.DataFrame:

    master_table = fte_df.merge(spine, on=["open_time", "close_time"], how="inner")
    assert master_table.shape[0] == fte_df.shape[0] == spine.shape[0], \
            "Mismatch of dates between features and spine, review."

    master_table_dropped = master_table.dropna()

    # drop useless columns (they're in spine layer if any troubleshooting needed)
    master_table_dropped = master_table_dropped.drop(columns=["target_time_log_return", "std"])
    master_table_numbered = _build_window_numbers(df=master_table_dropped)

    window_nbr_lookup = master_table_numbered[["window_nbr", "open_time", "close_time", "target_time"]]
    master_table_numbered = master_table_numbered.drop(columns=["open_time", "target_time",
                                                        "target_time_close", "close_time_close",
                                                        "close_to_tgt_time_logret", "logret_cumsum"])

    _check_master_table_quality(df=master_table_numbered, class_bounds=class_bounds)

    return master_table_numbered, window_nbr_lookup


def _build_window_numbers(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(by="close_time", ascending=True)
    df = df.reset_index(drop=True)
    idxs = [idx + 1 for idx in  df.index]

    df.loc[:, "window_nbr"] = idxs

    return df


def _check_master_table_quality(df: pd.DataFrame, class_bounds: Dict[str, float]) -> None:

    # check nulls
    assert df.isnull().sum().sum() == 0, "Master table contains null, review."

    # check label unbalancing
    labels_pct = (df.label.value_counts() / df.shape[0]).values
    assert any([(label_pct >= class_bounds["lower"] and \
                    label_pct <= class_bounds["upper"]) \
                for label_pct in labels_pct]), "Unbalanced classes, review."
