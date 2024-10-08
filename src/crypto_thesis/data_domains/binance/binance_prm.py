# -*- coding: utf-8 -*-
import logging
from datetime import timedelta
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def binance_prm(binance_raw: pd.DataFrame,
                min_years_existence: int,
                end_date: str) -> pd.DataFrame:
    """Standardize data acquired in the raw layer, so all other tasks can use the same data

    Args:
        binance_raw (pd.DataFrame): dataframe with raw data
        min_years_existence (int): minimum number, in years, for ticker existence
        end_date (str): upper bound date to start counting the lookback window

    Returns:
        pd.DataFrame: dataframe with standardized data
    """

    # in this case it's indifferent to get `open_time` or `close_time`
    binance_prm = binance_raw[["open_time", "open", "high", "low", "close", "volume", "symbol"]]
    binance_prm.loc[:, "open_time"] = pd.to_datetime(binance_prm["open_time"], unit="ms")

    # get symbols that do not pass the criteria and exclude them
    symbols_date_excluded = _find_symbol_date_criterium(df=binance_prm,
                                                        max_date=end_date,
                                                        min_years_existence=min_years_existence)
    binance_prm_excl = binance_prm[~binance_prm["symbol"].isin(symbols_date_excluded)]
    logger.info(f"Number of symbols excluded due to selection criteria: "\
                f"{binance_prm['symbol'].nunique() - binance_prm_excl['symbol'].nunique()}")

    # get all columns to cast as float
    _float_cols = [col for col in binance_prm_excl.columns if col not in ["open_time", "symbol"]]
    _float_cols_dict = {key: "float64" for key in _float_cols}
    binance_prm_excl = binance_prm_excl.astype(_float_cols_dict)

    return binance_prm_excl


def binance_prm_oos(binance_raw_oos: pd.DataFrame,
                    binance_prm_is: pd.DataFrame) -> pd.DataFrame:
    """Standardize data acquired in the raw layer, so all other tasks can use the same data

    Args:
        binance_raw (pd.DataFrame): dataframe with raw data
        min_years_existence (int): minimum number, in years, for ticker existence
        end_date (str): upper bound date to start counting the lookback window

    Returns:
        pd.DataFrame: dataframe with standardized data
    """

    # in this case it's indifferent to get `open_time` or `close_time`
    binance_prm = binance_raw_oos[["open_time", "open", "high", "low", "close", "volume", "symbol"]]
    binance_prm.loc[:, "open_time"] = pd.to_datetime(binance_prm["open_time"], unit="ms")

    binance_prm_excl = binance_prm[binance_prm["symbol"].isin(binance_prm_is["symbol"].unique())]
    logger.info(f"Number of symbols excluded due to selection criteria: "\
                f"{binance_prm['symbol'].nunique() - binance_prm_excl['symbol'].nunique()}")

    # get all columns to cast as float
    _float_cols = [col for col in binance_prm_excl.columns if col not in ["open_time", "symbol"]]
    _float_cols_dict = {key: "float64" for key in _float_cols}
    binance_prm_excl = binance_prm_excl.astype(_float_cols_dict)

    return binance_prm_excl

def _find_symbol_date_criterium(df: pd.DataFrame,
                                max_date: str,
                                min_years_existence: int) -> List[str]:
    """Find which tickers must be excluded due to dates' criteria violation

    Args:
        df (pd.DataFrame): dataframe with data to be validated
        max_date (str): upper bound date to start counting the lookback window
        min_years_existence (int): minimum number, in years, for ticker existence

    Returns:
        List[str]: list of tickers to be excluded
    """

    # generate min and max time windows
    df_aux = df.copy()
    df_aux.loc[:, "date"] = df_aux["open_time"].dt.date.apply(str)
    df_aux = df_aux.groupby("symbol").agg({"date": ["min", "max"]}).reset_index()
    df_aux.columns = ["symbol", "date_min", "date_max"]

    # build boundaries and parse values to same data type
    max_date = pd.to_datetime(max_date)
    cutoff_3y = str((max_date - timedelta(days = min_years_existence * 365)).date())
    max_date = str(max_date.date())

    # apply rule
    df_aux = df_aux[(df_aux["date_min"] >= cutoff_3y) | (df_aux["date_max"] < max_date)]

    return df_aux["symbol"].unique().tolist()
