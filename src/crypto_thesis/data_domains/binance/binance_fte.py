# -*- coding: utf-8 -*-
import warnings
from functools import reduce
from typing import Dict, List, Union

import numpy as np
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

    # accumulate data within the volume bar window
    for start, end in zip(spine_preproc["open_time"], spine_preproc["close_time"]):
        df_aux = df_log_ret[df_log_ret["open_time"].between(start, end)]
        _numeric_cols = df_aux.select_dtypes(include=np.float64).columns.tolist()

        dfs_agg = []
        for _col in _numeric_cols:
            df_aux_agg_ftes = _build_agg_ftes(df=df_aux, ref_col=_col, grpby_col="symbol")
            dfs_agg.append(df_aux_agg_ftes)

        df_agg_ftes = reduce(lambda left, right: pd.merge(left, right, on="symbol", how="inner"), dfs_agg)
        df_agg_ftes.loc[:, ["open_time", "close_time"]] = [start, end]

        df_ts = _build_timeseries(df=df_agg_ftes, index=["open_time", "close_time"], cols=["symbol"])
        final_df = pd.concat([final_df, df_ts])

    return final_df

def _build_agg_ftes(df: pd.DataFrame, ref_col: str, grpby_col: str) -> pd.DataFrame:
    """Improve this logic, what if we have 10 columns?
    """

    df_aux_accum = df.groupby(grpby_col)[ref_col].sum()
    df_aux_accum = df_aux_accum.reset_index().rename(columns={ref_col: f"{ref_col}_accum"})

    df_aux_mean = df.groupby(grpby_col)[ref_col].mean()
    df_aux_mean = df_aux_mean.reset_index().rename(columns={ref_col: f"{ref_col}_mean"})

    df_aux_std = df.groupby(grpby_col)[ref_col].std(ddof=1)
    df_aux_std = df_aux_std.reset_index().rename(columns={ref_col: f"{ref_col}_std"})

    df_aux = reduce(lambda left, right: pd.merge(left, right, on=grpby_col, how="inner"),
                    [df_aux_accum, df_aux_mean, df_aux_std])

    assert df_aux.shape[0] == df_aux_accum.shape[0] == df_aux_mean.shape[0] == df_aux_std.shape[0], \
            "Wrong data length, review."

    return df_aux

def _build_timeseries(df: pd.DataFrame, index: Union[str, List[str]], cols: List[str]) -> pd.DataFrame:

    df_pivot = df.pivot(index, cols)
    _single_index_cols = df_pivot.columns.map("__".join)

    df_pivot.columns = _single_index_cols
    df_pivot = df_pivot.reset_index()

    return df_pivot
