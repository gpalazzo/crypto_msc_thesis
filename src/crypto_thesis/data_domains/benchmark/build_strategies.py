import pandas as pd
import numpy as np


def buy_and_hold_strategy(df_window_nbr: pd.DataFrame,
                          df_px: pd.DataFrame,
                          target_name: str) -> pd.DataFrame:

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