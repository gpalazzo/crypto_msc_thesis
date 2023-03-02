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
    bars_ahead_predict = preproc_params["bar_ahead_predict"]

    preproc_df = prm_binance[prm_binance["symbol"] == _target_name]
    # don't change this df structure because it will be used later on
    preproc_df = preproc_df[["close_time", "close", "volume"]]

    if preproc_df.empty:
        raise RuntimeError(f"Target name {_target_name} doesn't have any data.")

    # if the min volume is less than the volume bar size, then we can't accumulate it
    # therefore, every data point is gonna be a predict point which isn't good for our use case
    if preproc_df["volume"].min() > _volume_bar_size:
        raise RuntimeError("Specified volume bar size isn't correct, please review.")

    preproc_df = preproc_df.sort_values(by="close_time")
    df_log_ret = build_log_return(df=preproc_df)
    # fill first null data point with 0 to avoid having NaN at first volume window
    df_log_ret.loc[:, "log_return"] = df_log_ret["log_return"].fillna(0)

    # TODO: should the log return in each window open_time start with 0? This way we would treat
    # each window as a new entity without any relation to data outside the window range
    df, idxs = _build_threshold_flag(preproc_df=df_log_ret, _volume_bar_size=_volume_bar_size)
    df = _build_flag_time_window(df=df, idxs=idxs, bars_ahead=bars_ahead_predict)
    df = _get_target_time_values(df=df, bars_ahead=bars_ahead_predict)

    # get useful columns for calculation in labeling
    df = df[["open_time", "close_time", "target_time", "logret_cumsum", "target_time_close", "target_time_log_return"]]
    df_tgt_px = df.merge(preproc_df[["close_time", "close"]] \
                            .rename(columns={"close": "close_time_close"}),
                        on="close_time",
                        how="inner")

    assert df_tgt_px.shape[0] == df.shape[0], "Data loss when joining to get close_time price, review."

    return df_tgt_px, df_log_ret[["close_time", "log_return"]]


def _build_threshold_flag(preproc_df: pd.DataFrame, _volume_bar_size: Union[int, float]) -> pd.DataFrame:

    preproc_df = preproc_df.sort_values(by="close_time", ascending=True)

    volume_ls = []
    logret_ls = []
    idxs = []
    volume_cumsum = 0
    logret_cumsum = 0

    for i, row in preproc_df.iterrows():
        if volume_cumsum + row.volume <= _volume_bar_size:
            volume_cumsum += row.volume
            logret_cumsum += row.log_return
        else:
            idxs.append(i - 1) #when it goes here, it means the last index reach the `_volume_bar_size`
            volume_cumsum = row.volume
            logret_cumsum = row.log_return
        volume_ls.append(volume_cumsum)
        logret_ls.append(logret_cumsum)

    preproc_df["volume_cumsum"] = volume_ls
    preproc_df["logret_cumsum"] = logret_ls
    preproc_df.loc[idxs, "reach_threshold"] = True

    return preproc_df, idxs


def _build_flag_time_window(df: pd.DataFrame, idxs: List[int], bars_ahead: int) -> pd.DataFrame:

    final_df = pd.DataFrame()

    for i, idx in enumerate(idxs):

        if i == 0:
            _open_time = df.iloc[0].close_time
        else:
            _target_idx = idxs[i-1] + bars_ahead
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


def _get_target_time_values(df: pd.DataFrame, bars_ahead: int) -> pd.DataFrame:

    df = df.sort_values(by="close_time")
    df.loc[:, "next_log_return"] = df["log_return"].shift(-bars_ahead)
    df.loc[:, "next_close"] = df["close"].shift(-bars_ahead)

    # remove last data point
    df_drop = df[df["next_log_return"].notnull()]
    assert df_drop.shape[0] == df.shape[0] - 1, "More than 1 data point was removed, review."

    df_drop = df_drop.drop(columns=["log_return"]).rename(columns={"next_log_return": "target_time_log_return",
                                                        "next_close": "target_time_close"})

    return df_drop
