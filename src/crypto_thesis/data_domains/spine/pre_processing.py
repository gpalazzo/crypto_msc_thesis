# -*- coding: utf-8 -*-
from typing import Dict, List, Union

import pandas as pd

from crypto_thesis.utils import build_log_return


def spine_preprocessing(prm_binance: pd.DataFrame, preproc_params: Dict[str, str]) -> pd.DataFrame:

    # crucial step because the dataframe index will be used in this step
    prm_binance = prm_binance.reset_index(drop=True)
    prm_binance = prm_binance.rename(columns={"open_time": "close_time"})

    _target_name = preproc_params["target_name"]
    _volume_bar_size = preproc_params["volume_bar_size"]

    preproc_df = prm_binance[prm_binance["symbol"] == _target_name]
    preproc_df = preproc_df[["close_time", "close", "volume"]]

    if preproc_df.empty:
        raise RuntimeError(f"Target name {_target_name} doesn't have any data.")

    # given the fact it's a daily dataframe, if the minimum or maximum volume is higher than the
        # selected threshold, then we would need intraday data to build the volume bar size
        # therefore, let's break the code for now
    if preproc_df["volume"].min() > _volume_bar_size or preproc_df["volume"].max() > _volume_bar_size:
        raise RuntimeError("Specified volume bar size isn't correct, please review.")

    df = build_log_return(df=preproc_df)

    df, idxs = _build_threshold_flag(preproc_df=df, _volume_bar_size=_volume_bar_size)
    df = _build_flag_time_window(df=df, idxs=idxs)

    df = df[["open_time", "close_time", "target_time", "log_return"]]

    return df


def _build_threshold_flag(preproc_df: pd.DataFrame, _volume_bar_size: Union[int, float]) -> pd.DataFrame:

    preproc_df = preproc_df.sort_values(by="close_time", ascending=True)

    ls = []
    idxs = []
    cumsum = 0

    for i, row in preproc_df.iterrows():
        if cumsum + row.volume <= _volume_bar_size:
            cumsum += row.volume
        else:
            idxs.append(i - 1) #when it goes here, it means the last index reach the `_volume_bar_size`
            cumsum = row.volume
        ls.append(cumsum)

    preproc_df["cumsum"] = ls
    preproc_df.loc[idxs, "reach_threshold"] = True

    return preproc_df, idxs


def _build_flag_time_window(df: pd.DataFrame, idxs: List[int]) -> pd.DataFrame:

    final_df = pd.DataFrame()

    for i, idx in enumerate(idxs):

        if i == 0:
            _open_time = df.iloc[0].close_time
        else:
            _target_idx = idxs[i-1] + 1
            _open_time = df.iloc[_target_idx].close_time

        try:
            _future_idx = idxs[i+1]
            _target_time = df.iloc[_future_idx].close_time
        except IndexError:
            _target_time = None

        df_aux = df.filter(items=[idx], axis=0)
        df_aux.loc[:, "open_time"] = _open_time
        df_aux.loc[:, "target_time"] = _target_time

        final_df = pd.concat([final_df, df_aux])

    # remove last data point
    final_df = final_df[final_df["target_time"].notnull()]

    return final_df
