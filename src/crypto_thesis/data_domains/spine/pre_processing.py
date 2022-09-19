# -*- coding: utf-8 -*-
from typing import Dict

import pandas as pd


def spine_preprocessing(prm_binance: pd.DataFrame, preproc_params: Dict[str, str]) -> pd.DataFrame:

    _target_name = preproc_params["target_name"]
    _volume_bar_size = preproc_params["volume_bar_size"]

    preproc_df = prm_binance[prm_binance["symbol"] == _target_name]
    preproc_df = preproc_df[["open_time", "close", "volume"]]

    if preproc_df.empty:
        raise RuntimeError(f"Target name {_target_name} doesn't have any data.")

    # given the fact it's a daily dataframe, if the minimum or maximum volume is higher than the
        # selected threshold, then we would need intraday data to build the volume bar size
        # therefore, let's break the code for now
    if preproc_df["volume"].min() > _volume_bar_size or preproc_df["volume"].max() > _volume_bar_size:
        raise RuntimeError("Specified volume bar size isn't correct, please review.")

    preproc_df = preproc_df.sort_values(by="open_time", ascending=True)

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

    # TODO: criar lógica para fazer janela baseada no valor `True` da coluna `reach_threshold`
