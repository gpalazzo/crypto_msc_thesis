# -*- coding: utf-8 -*-
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_portfolio_pnl(df_predict: pd.DataFrame,
                window_lookup: pd.DataFrame,
                prices_df: pd.DataFrame,
                target_name: str
                ) -> pd.DataFrame:

    window_lookup = window_lookup.set_index("window_nbr")
    df = df_predict.merge(window_lookup, left_index=True, right_index=True, how="inner")
    df = df.sort_index().drop(columns=["open_time"]) #open_time col not useful for now

    assert df.shape[0] == df_predict.shape[0]

    # buy actions will only be taken when y_pred = 1 by construction
    # y_pred = 1 means the price level will be top compared to current market state
    df_top = df[df["y_pred"] == 1]

    df_target_prices = prices_df[prices_df["symbol"] == target_name][["open_time", "close"]]

    # get close time price
    df_merged_prices = df_top.merge(df_target_prices,
                                left_on="close_time",
                                right_on="open_time",
                                how="inner") \
                            .rename(columns={"close": "close_time_price"}) \
                            .drop(columns=["open_time"])
    assert df_top.shape[0] == df_merged_prices.shape[0], "Data loss when joining predictions with prices"

    # get target time price
    df_merged_prices = df_merged_prices.merge(df_target_prices,
                                            left_on="target_time",
                                            right_on="open_time",
                                            how="inner") \
                            .rename(columns={"close": "target_time_price"}) \
                            .drop(columns=["open_time"])

    df_merged_prices.loc[:, "operation_profit"] = df_merged_prices["target_time_price"] - \
                                                 df_merged_prices["close_time_price"]

    logger.info(f"***** Portfolio profit: {df_merged_prices['operation_profit'].sum()}")

    return df_merged_prices

def build_portfolio_metrics(df: pd.DataFrame) -> pd.DataFrame:
    pass
