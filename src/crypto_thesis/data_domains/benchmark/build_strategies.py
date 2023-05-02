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


def trend_following_strategy(spine_preproc: pd.DataFrame,
                             df_window_nbr: pd.DataFrame) -> pd.DataFrame:
    
    spine_preproc = spine_preproc[["open_time", "close_time", "close_time_close"]] \
                                .rename(columns={"close_time_close": "close_px"})
    
    spine_preproc.loc[:, "prev2_close_px"] = spine_preproc["close_px"].shift(2)
    spine_preproc.loc[:, "prev_close_px"] = spine_preproc["close_px"].shift()
    df_drop = spine_preproc.dropna() #drop first data point due to shift null
    assert df_drop.shape[0] == spine_preproc.shape[0] - 2, "More than 2 data points were dropped, review"

    df_drop.loc[:, "y_pred"] = df_drop.apply(lambda col: 1 \
                                            if col["prev2_close_px"] <= col["prev_close_px"] <= col["close_px"] \
                                            else 0 \
                                        , axis=1)

    df = df_drop.merge(df_window_nbr, 
                             on=["open_time", "close_time"], 
                             how="inner")
    assert df.shape[0] == df_window_nbr.shape[0], "Mismatch between spine preproc and window numbers"

    df = df.set_index("window_nbr")[["y_pred"]]

    return df