import pandas as pd
import numpy as np
from crypto_thesis.utils import build_log_return, build_timeseries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


def pc1_index_strategy(window_nbr: pd.DataFrame,
                       binance_prm: pd.DataFrame) -> pd.DataFrame:

    binance_prm = binance_prm.sort_values(by=["symbol", "open_time"])
    df_log_ret = binance_prm.groupby("symbol").apply(build_log_return)

    df_log_ret.loc[:, "pctchg"] = df_log_ret \
                                    .groupby("symbol")["log_return"] \
                                    .apply(lambda row: np.exp(row) - 1)
    df_log_ret = df_log_ret[["open_time", "symbol", "pctchg"]]
    df_ts = build_timeseries(df=df_log_ret, index=["open_time"], cols=["symbol"])
    df_ts = df_ts.dropna()

    for start, end in zip(window_nbr["open_time"], window_nbr["close_time"]):
        df_aux = df_ts[df_ts["open_time"].between(start, end)].set_index("open_time")
        cols = df_aux.columns

        scaler = StandardScaler()
        df_aux = pd.DataFrame(scaler.fit_transform(df_aux), columns=cols)
        pca = PCA(n_components=1)
        pca.fit(df_aux)
        comps = pd.DataFrame(pca.components_, columns=cols)

        comps_norm = comps / comps.values.sum()
        comps_norm.columns = comps_norm.columns.str.replace("pctchg", "weight")
        comps_norm.loc[:, ["open_time", "close_time"]] = start, end
        breakpoint()