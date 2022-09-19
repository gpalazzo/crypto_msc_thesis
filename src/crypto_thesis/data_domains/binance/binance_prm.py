# -*- coding: utf-8 -*-
import pandas as pd


def binance_prm(binance_raw: pd.DataFrame) -> pd.DataFrame:

    # in this case it's indifferent to get `open_time` or `close_time`
    binance_prm = binance_raw[["open_time", "open", "high", "low", "close", "volume", "symbol"]]

    binance_prm.loc[:, "open_time"] = pd.to_datetime(binance_prm["open_time"], unit="ms")

    # get all columns to cast as float
    _float_cols = [col for col in binance_prm.columns if col not in ["open_time", "symbol"]]
    _float_cols_dict = {key: "float64" for key in _float_cols}

    binance_prm = binance_prm.astype(_float_cols_dict)

    return binance_prm
