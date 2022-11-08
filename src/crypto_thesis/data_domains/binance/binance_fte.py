# -*- coding: utf-8 -*-
import warnings
from typing import Dict, List

import pandas as pd

from crypto_thesis.utils import build_log_return

warnings.filterwarnings("ignore")


def binance_fte(binance_prm: pd.DataFrame, spine_preproc: pd.DataFrame, spine_params: Dict[str, str]) -> pd.DataFrame:

    final_df = pd.DataFrame()

    _target_name = spine_params.get("target_name")
    binance_prm = binance_prm[binance_prm["symbol"] != _target_name]
    binance_prm = binance_prm.sort_values(by=["symbol", "open_time"])

    df_log_ret = build_log_return(df=binance_prm)
    df_log_ret = df_log_ret[["open_time", "symbol", "log_return", "volume"]]

    for start, end in zip(spine_preproc["open_time"], spine_preproc["close_time"]):
        df_aux = df_log_ret[df_log_ret["open_time"].between(start, end)]
        df_aux.loc[:, "mean_10_std"] = df_aux.groupby("symbol")["log_return"].rolling(10).mean(skip_na=True).values
        df_aux.loc[:, "roll_10_std"] = df_aux.groupby("symbol")["log_return"].rolling(10).std(ddof=1, skip_na=True).values
        df_aux = _build_zscore(df=df_aux, ref_col="volume", window=10, grpby_col="symbol")

        df_ts = _build_timeseries(df=df_aux, index="open_time", cols=["symbol"])
        # rename timestamp to not overwrite within the boundaries
        df_ts = df_ts.rename(columns={"open_time": "timestamp"})
        df_ts.loc[:, ["open_time", "close_time"]] = [start, end]

        df_dropna = df_ts.dropna()

        final_df = pd.concat([final_df, df_dropna])

    return final_df


def _build_zscore(df: pd.DataFrame, ref_col: str, window: int, grpby_col: str) -> pd.DataFrame:

    df.loc[:, "mean"] = df.groupby(grpby_col)[ref_col].rolling(window).mean(skip_na=True).values
    df.loc[:, "std"] = df.groupby(grpby_col)[ref_col].rolling(window).std(ddof=1, skip_na=True).values
    df.loc[:, f"{ref_col}_zscore"] = (df["volume"] - df["mean"]) / df["std"]

    return df.drop(columns=["mean", "std"])

def _build_timeseries(df: pd.DataFrame, index: str, cols: List[str]) -> pd.DataFrame:

    df_pivot = df.pivot(index, cols)
    _single_index_cols = df_pivot.columns.map("__".join)

    df_pivot.columns = _single_index_cols
    df_pivot = df_pivot.reset_index()

    return df_pivot
