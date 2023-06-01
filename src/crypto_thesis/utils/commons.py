# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Union, List


def build_log_return(df: pd.DataFrame, ref_col: str = "close") -> pd.DataFrame:
    """Build log return from prices. It assumes there's only one asset in the dataframe

    Args:
        df (pd.DataFrame): dataframe with prices
        ref_col (str, optional): column name having the prices. Defaults to "close".

    Returns:
        pd.DataFrame: dataframe with prices and log-returns
    """

    df.loc[:, "shift"] = df[ref_col].shift()
    df.loc[:, "log_return"] = np.log(df[ref_col] / df["shift"])

    return df


def build_timeseries(df: pd.DataFrame, index: Union[str, List[str]], cols: List[str]) -> pd.DataFrame:
    """Build timeseries, i.e. one value per date, by pivotting dataframe. It recreates the pivotted columns 
    by separating the names with double underscores (__)

    Args:
        df (pd.DataFrame): dataframe to be pivotted
        index (Union[str, List[str]]): column names to be used as pivot index
        cols (List[str]): column names to be pivotted

    Returns:
        pd.DataFrame: dataframe with pivotted columns having names separated by double underscores (__)
    """

    df_pivot = df.pivot(index, cols)
    _single_index_cols = df_pivot.columns.map("__".join)

    df_pivot.columns = _single_index_cols
    df_pivot = df_pivot.reset_index()

    return df_pivot