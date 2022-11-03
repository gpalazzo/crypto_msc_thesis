# -*- coding: utf-8 -*-
from typing import Dict

import pandas as pd


def binance_fte(binance_prm: pd.DataFrame, spine_preproc: pd.DataFrame, spine_params: Dict[str, str]) -> pd.DataFrame:

    _target_name = spine_params.get("target_name")
    binance_prm = binance_prm[binance_prm["symbol"] != _target_name]
    binance_prm = binance_prm.sort_values(by=["symbol", "open_time"])

    for start, end in zip(spine_preproc["open_time"], spine_preproc["close_time"]):
        df_aux = binance_prm[binance_prm["open_time"].between(start, end)]
        # breakpoint()
        df_aux.loc[:, "min_open"] = df_aux.groupby("symbol")["open"].rolling(window = 5).min()
