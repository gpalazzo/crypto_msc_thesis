# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def build_log_return(df: pd.DataFrame, ref_col: str = "close") -> pd.DataFrame:

    df = df.sort_values(by="close_time")

    df.loc[:, "shift"] = df[ref_col].shift()
    df.loc[:, "log_return"] = np.log(df[ref_col] / df["shift"])

    return df[["close_time", "log_return", "volume"]]
