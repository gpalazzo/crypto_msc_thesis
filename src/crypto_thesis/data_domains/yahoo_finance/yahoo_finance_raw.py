# -*- coding: utf-8 -*-
from datetime import timedelta
from typing import Any, Dict

import pandas as pd
import yfinance as yf


def yahoo_finance_raw(raw_yahoo_finance_get_data: Dict[str, Any]) -> pd.DataFrame:

    tickers = list(raw_yahoo_finance_get_data["tickers"].keys())

    # sum 1 day to end_date because yahoo finance is not inclusive in the end date
    _end_date = pd.to_datetime(raw_yahoo_finance_get_data["end_date"])
    _end_date_plusone = _end_date + timedelta(days=1)
    _end_date_plusone_fmt = str(_end_date_plusone.date())

    df = yf.download(
        tickers=tickers,
        start=raw_yahoo_finance_get_data["start_date"],
        end=_end_date_plusone_fmt,
        interval="1d",
        auto_adjust=True #it automatically adjust prices for corporate events
    )

    df = df.stack().reset_index().rename(columns={"level_0": "date", "level_1": "ticker"})

    assert (
        len(raw_yahoo_finance_get_data["tickers"]) == df["ticker"].nunique()
    ), "Some of the tickers we're not found, please review"

    return df
