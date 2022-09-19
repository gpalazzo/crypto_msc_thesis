# -*- coding: utf-8 -*-
import logging
import os
from typing import Any, Dict

import pandas as pd
from binance.client import Client

logger = logging.getLogger(__name__)


def binance_raw(raw_binance_get_data: Dict[str, Any]) -> pd.DataFrame:

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    client = Client(api_key, api_secret)

    interval = "1d"
    Client.KLINE_INTERVAL_1DAY

    final_df = pd.DataFrame()

    _total_tickers = len(raw_binance_get_data["tickers"])

    for i, symbol in enumerate(raw_binance_get_data["tickers"], 1):

        logging.info(f"Pulling data for {symbol}... {i} out of {_total_tickers}")

        klines = client.get_historical_klines(symbol, interval, raw_binance_get_data["start_date"])
        data = pd.DataFrame(klines)

        # create colums name
        data.columns = ["open_time","open", "high", "low", "close", "volume","close_time", "qav","num_trades","taker_base_vol", "taker_quote_vol", "ignore"]
        data.loc[:, "symbol"] = symbol

        final_df = pd.concat([final_df, data])

    assert final_df["symbol"].nunique() == _total_tickers, "Not all tickers were collected, please review"

    return final_df
