# -*- coding: utf-8 -*-
from typing import Dict

import numpy as np
import pandas as pd


def yahoo_finance_prm(yf_raw: pd.DataFrame, map_names_params: Dict[str, str]) -> pd.DataFrame:

    map_ticker_names = map_names_params.get("tickers")

    yf_raw.loc[:, "ticker"] = yf_raw["ticker"].map(map_ticker_names)
    assert yf_raw["ticker"].isna().sum() == 0, "Missing ticker mapping, review."

    yf_raw.columns = yf_raw.columns.str.lower()

    df_prm = _calculate_log_return(df=yf_raw)

    return df_prm


def _calculate_log_return(df: pd.DataFrame, ref_col_name: str = "close") -> pd.DataFrame:

    grpby_col = "ticker"
    sort_col = "date"

    df_sorted = df.sort_values(by=[grpby_col, sort_col])

    df_sorted.loc[:, f"{ref_col_name}_pctchange"] = df_sorted.groupby(grpby_col)[ref_col_name].pct_change()
    df_sorted.loc[:, f"{ref_col_name}_logret"] = np.log(1 + df_sorted[f"{ref_col_name}_pctchange"])

    df_sorted = df_sorted.drop(columns=[f"{ref_col_name}_pctchange"])

    # these manipulations must not change dataframe size
    assert df.shape[0] == df_sorted.shape[0], "Wrong manipulation changed df size."

    return df_sorted
