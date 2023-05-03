# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Union, List


def build_log_return(df: pd.DataFrame, ref_col: str = "close") -> pd.DataFrame:

    df.loc[:, "shift"] = df[ref_col].shift()
    df.loc[:, "log_return"] = np.log(df[ref_col] / df["shift"])

    return df


def build_timeseries(df: pd.DataFrame, index: Union[str, List[str]], cols: List[str]) -> pd.DataFrame:

    df_pivot = df.pivot(index, cols)
    _single_index_cols = df_pivot.columns.map("__".join)

    df_pivot.columns = _single_index_cols
    df_pivot = df_pivot.reset_index()

    return df_pivot