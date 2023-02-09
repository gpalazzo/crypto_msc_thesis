# -*- coding: utf-8 -*-
import pandas as pd


def build_master_table(fte_df: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:

    master_table = fte_df.merge(spine, on=["open_time", "close_time"], how="inner")
    master_table = master_table.dropna()

    assert master_table.isnull().sum().sum() == 0, "Master table contains null, review."

    master_table_dropped = _check_drop_rows_qty_byWindow(df=master_table)

    print(f"Total rows dropped: {master_table.shape[0] - master_table_dropped.shape[0]}")

    # drop useless columns (they're in spine layer if any troubleshooting needed)
    master_table_dropped = master_table_dropped.drop(columns=["timestamp", "log_return", "std"])

    return master_table_dropped


def _check_drop_rows_qty_byWindow(df: pd.DataFrame) -> pd.DataFrame:

    df = df.reset_index(drop=True)

    for start, end in zip(df["open_time"], df["close_time"]):
        df_aux = df[(df["open_time"] == start) & (df["close_time"] == end)]

        # drop interval if the numbers of rows is less than the numbers of columns * 5
        # it guarantees the number of rows is greater than the columns by an arbitrary amount
        if df_aux.shape[1] * 5 > df_aux.shape[0]:
            df = df.drop(df_aux.index)

    return df
