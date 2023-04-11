# -*- coding: utf-8 -*-
import logging
import warnings
from functools import reduce
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import stats

from crypto_thesis.utils import build_log_return

warnings.filterwarnings("ignore")

# column name to differ securities to apply transformations
# it will have no effect after the data becomes a time series
IDENTIFIER_COL = "symbol"
INDEX_COL = ["open_time", "close_time"]

logger = logging.getLogger(__name__)


def binance_fte(binance_prm: pd.DataFrame,
                spine_labeled: pd.DataFrame,
                spine_params: Dict[str, str]) -> pd.DataFrame:

    final_df = pd.DataFrame()

    binance_prm = binance_prm[binance_prm[IDENTIFIER_COL] != spine_params.get("target_name")]
    binance_prm = binance_prm.sort_values(by=[IDENTIFIER_COL, "open_time"])

    df_log_ret = binance_prm.groupby(IDENTIFIER_COL).apply(build_log_return)
    df_log_ret.loc[:, "pctchg"] = df_log_ret \
                                    .groupby(IDENTIFIER_COL)["log_return"] \
                                    .apply(lambda row: np.exp(row) - 1)
    df_log_ret = df_log_ret[["open_time", IDENTIFIER_COL, "pctchg", "log_return", "volume"]]

    # accumulate data within the volume bar window
    for start, end in zip(spine_labeled["open_time"], spine_labeled["close_time"]):
        df_aux = df_log_ret[df_log_ret["open_time"].between(start, end)]

        df_aux = _null_handler(df=df_aux)

        df_agg = _build_agg_ftes(df=df_aux, grpby_col=IDENTIFIER_COL, window_start=start, window_end=end)
        df_biz_fte = _build_business_ftes(df=df_aux, window_start=start, window_end=end)

        df_ftes = reduce(lambda left, right: pd.merge(left, right,
                                                    on=["open_time", "close_time", IDENTIFIER_COL],
                                                    how="inner"),
                    [df_agg, df_biz_fte])

        df_ts = _build_timeseries(df=df_ftes, index=["open_time", "close_time"], cols=[IDENTIFIER_COL])

        # last feature: only dependant on the window size, regardless of the amount of securities
        df_ts.loc[:, "window_duration_sec"] = (end - start).total_seconds()
        final_df = pd.concat([final_df, df_ts])

    return final_df


def _null_handler(df: pd.DataFrame) -> pd.DataFrame:

    df.loc[:, ["pctchg", "log_return"]] = df[["pctchg", "log_return"]].fillna(0)
    assert df.isna().sum().sum() == 0, "There are null values even after null handling, review."

    return df

def _build_agg_ftes(df: pd.DataFrame,
                    grpby_col: str,
                    window_start: pd.Timestamp,
                    window_end: pd.Timestamp) -> pd.DataFrame:
    """Improve this logic, what if we have 10 columns?
    """

    def __aggregator(df, ref_col, grpby_col):
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


    df = df.sort_values(by=[IDENTIFIER_COL, "open_time"])
    _numeric_cols = df.select_dtypes(include=np.float64).columns.tolist()

    dfs_agg = []
    for _col in _numeric_cols:
        df_aux_agg_ftes = __aggregator(df=df, ref_col=_col, grpby_col=grpby_col)
        dfs_agg.append(df_aux_agg_ftes)

    df_agg_ftes = reduce(lambda left, right: pd.merge(left, right, on=IDENTIFIER_COL, how="inner"), dfs_agg)
    df_agg_ftes.loc[:, ["open_time", "close_time"]] = [window_start, window_end]

    # adjust columns to add pctchg and remove logret
    df_agg_ftes.loc[:, "pctchg_accum"] = np.exp(df_agg_ftes["log_return_accum"]) - 1
    _cols = [col for col in df_agg_ftes.columns if col.startswith("log_return_")]
    df_agg_ftes = df_agg_ftes.drop(columns=_cols)

    return df_agg_ftes


def _build_business_ftes(df: pd.DataFrame,
                        window_start: pd.Timestamp,
                        window_end: pd.Timestamp) -> pd.DataFrame:

    def __get_last_zscore(df: pd.DataFrame) -> float:
        df.loc[:, "zscore"] = df.groupby(IDENTIFIER_COL)["pctchg"].transform(lambda x:
                                                                    stats.zscore(x,
                                                                        axis=0,
                                                                        ddof=1,
                                                                        nan_policy="omit")
                                                                    )
        df = df.groupby(IDENTIFIER_COL)["zscore"].last().reset_index()

        return df

    df = df.sort_values(by=[IDENTIFIER_COL, "open_time"])
    df_zscore = __get_last_zscore(df=df)

    df_zscore.loc[:, ["open_time", "close_time"]] = [window_start, window_end]

    return df_zscore


def _build_timeseries(df: pd.DataFrame, index: Union[str, List[str]], cols: List[str]) -> pd.DataFrame:

    df_pivot = df.pivot(index, cols)
    _single_index_cols = df_pivot.columns.map("__".join)

    df_pivot.columns = _single_index_cols
    df_pivot = df_pivot.reset_index()

    return df_pivot
