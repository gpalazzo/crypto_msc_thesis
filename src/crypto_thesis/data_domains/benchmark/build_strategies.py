# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from crypto_thesis.utils import mt_split_train_test

abc = ""

TARGET_COL = ["label"]
# these cols were useful so far, but not anymore
INDEX_COL = "window_nbr"


def buy_and_hold_strategy(df_window_nbr: pd.DataFrame,
                          df_px: pd.DataFrame,
                          target_name: str) -> pd.DataFrame:
    """Create positions based on buy and hold strategy

    Args:
        df_window_nbr (pd.DataFrame): dataframe with start and end timestamps to be considered for the strategy
        df_px (pd.DataFrame): dataframe with prices
        target_name (str): name of the target coin

    Returns:
        pd.DataFrame: dataframe with defined positions for the strategy
    """

    # get prices
    df_tgt_px = df_px[df_px["symbol"] == target_name][["open_time", "close", "symbol"]]
    assert df_tgt_px["symbol"].unique()[0] == target_name

    # get times
    trade_start = df_window_nbr["open_time"].min()
    trade_end = df_window_nbr["close_time"].max()
    df_trades = df_tgt_px[df_tgt_px["open_time"].between(trade_start, trade_end)]

    df_trades = df_trades.sort_values(by="open_time", ascending=True)
    df_trades.loc[:, "pctchg"] = df_trades["close"].pct_change()
    df_trades.loc[:, "log_return"] = np.log(1 + df_trades["pctchg"])

    df_trades = df_trades.drop(columns=["symbol"]) \
                        .rename(columns={"open_time": "date", "close": "close_px"})

    return df_trades


def trend_following_strategy(spine_preproc: pd.DataFrame,
                             df_window_nbr: pd.DataFrame,
                             train_test_cutoff_date: str,
                             master_table: pd.DataFrame) -> pd.DataFrame:
    """Create positions based on trend following strategy

    Args:
        spine_preproc (pd.DataFrame): dataframe with pre-processed data for spine
        df_window_nbr (pd.DataFrame): dataframe with start and end timestamps to be considered for the strategy
        train_test_cutoff_date (str): date to cutoff dataset into train and test
        master_table (pd.DataFrame): dataframe with master table data

    Returns:
        pd.DataFrame: dataframe with defined positions for the strategy
    """

    _, _, X_test, _ = mt_split_train_test(master_table=master_table,
                                                            index_col=INDEX_COL,
                                                            train_test_cutoff_date=train_test_cutoff_date,
                                                            target_col=TARGET_COL)

    spine_preproc = spine_preproc[["open_time", "close_time", "logret_cumsum"]]
    df = spine_preproc.merge(df_window_nbr,
                             on=["open_time", "close_time"],
                             how="inner")
    df = df[df["window_nbr"].isin(X_test.index)]
    assert df.shape[0] == len(X_test.index), "Mismatch between spine preproc, window numbers and X_test index"

    df = df.drop(columns=["open_time", "close_time", "target_time"]).set_index("window_nbr").sort_index()

    df.loc[:, "prev_logret_cumsum"] = df["logret_cumsum"].shift()
    df_drop = df.dropna() #drop first data point due to shift null
    assert df_drop.shape[0] == df.shape[0] - 1, "More than 1 data point was dropped, review"

    df_drop.loc[:, "y_pred"] = df_drop.apply(lambda col: 1 \
                                            if col["logret_cumsum"] > col["prev_logret_cumsum"] \
                                            else 0 \
                                        , axis=1)

    df = df_drop[["y_pred"]]

    return df
