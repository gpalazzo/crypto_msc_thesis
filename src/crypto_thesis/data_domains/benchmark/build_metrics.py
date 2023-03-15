# -*- coding: utf-8 -*-
import math
from typing import Union

import pandas as pd
import quantstats as qs


def build_benchmark_metrics(df_benchmark: pd.DataFrame,
                df_predict: pd.DataFrame,
                window_lookup: pd.DataFrame,
                portfolio_initial_money: Union[int, float]) -> pd.DataFrame:

    window_lookup = window_lookup.set_index("window_nbr")
    df = df_predict.merge(window_lookup, left_index=True, right_index=True, how="inner")
    df = df.sort_index()

    assert df.shape[0] == df_predict.shape[0]

    # buy actions will only be taken when y_pred = 1 by construction
    # y_pred = 1 means the price level will be top compared to current market state
    df_top = df[df["y_pred"] == 1]

    df_pnl = _build_benchmark_pnl(df_top=df_top,
                                prices_df=df_benchmark,
                                portfolio_initial_money=portfolio_initial_money)

    df_perf_metr = _build_perf_metrics(df_pnl=df_pnl)

    return df_pnl, df_perf_metr


def _build_benchmark_pnl(df_top: pd.DataFrame,
                            prices_df: pd.DataFrame,
                            portfolio_initial_money: Union[int, float]) -> pd.DataFrame:
    """Assumption: since we don't have intraday prices, let's get the dates with trades (label = 1)
    and collect the benchmark prices. Then let's assume each subsequential is a target time
    """

    prices_df = prices_df.drop(columns=["pctchg", "log_return"])
    prices_df = prices_df.rename(columns={"date": "close_time", "close_px": "close_time_price"})
    prices_df.loc[:, "close_time"] = prices_df["close_time"].dt.date

    df_top.loc[:, "close_time"] = df_top["close_time"].dt.date
    df_top = df_top[["close_time"]].drop_duplicates()

    # get close time price
    df_merged_prices = df_top.merge(prices_df,on="close_time",how="inner")
    df_merged_prices = df_merged_prices.sort_values(by="close_time").reset_index(drop=True)

    df_merged_prices.loc[:, "target_time"] = df_merged_prices["close_time"].shift(-1)
    df_merged_prices.loc[:, "target_time_price"] = df_merged_prices["close_time_price"].shift(-1)
    df_merged_prices = df_merged_prices.dropna()

    df_merged_prices.loc[:, "op_unit_profit"] = df_merged_prices["target_time_price"] - \
                                                 df_merged_prices["close_time_price"]

    for i, row in df_merged_prices.iterrows():
        if i == 0: #first trade
            df_merged_prices.loc[i:i, "stock_qty"] = math.floor(portfolio_initial_money / row.close_time_price)
            df_merged_prices.loc[i:i, "op_full_profit"] = df_merged_prices.iloc[i]["stock_qty"] * df_merged_prices.iloc[i]["op_unit_profit"]
            df_merged_prices.loc[i:i, "buy_nominal_pos"] = df_merged_prices.iloc[i]["stock_qty"] * df_merged_prices.iloc[i]["close_time_price"]
            df_merged_prices.loc[i:i, "sell_nominal_pos"] = df_merged_prices.iloc[i]["stock_qty"] * df_merged_prices.iloc[i]["target_time_price"]
            df_merged_prices.loc[i:i, "residual_value"] = portfolio_initial_money - df_merged_prices.iloc[i]["buy_nominal_pos"]
            df_merged_prices.loc[i:i, "pctchg_pos"] = df_merged_prices.iloc[i]["sell_nominal_pos"] / df_merged_prices.iloc[i]["buy_nominal_pos"] - 1
        else:
            _total_curr_money = df_merged_prices.iloc[i-1].sell_nominal_pos + df_merged_prices.iloc[i-1].residual_value
            df_merged_prices.loc[i:i, "stock_qty"] = math.floor(_total_curr_money / row.close_time_price)
            df_merged_prices.loc[i:i, "op_full_profit"] = df_merged_prices.iloc[i]["stock_qty"] * df_merged_prices.iloc[i]["op_unit_profit"]
            df_merged_prices.loc[i:i, "buy_nominal_pos"] = df_merged_prices.iloc[i]["stock_qty"] * df_merged_prices.iloc[i]["close_time_price"]
            df_merged_prices.loc[i:i, "sell_nominal_pos"] = df_merged_prices.iloc[i]["stock_qty"] * df_merged_prices.iloc[i]["target_time_price"]
            df_merged_prices.loc[i:i, "residual_value"] = _total_curr_money - df_merged_prices.iloc[i]["buy_nominal_pos"]
            df_merged_prices.loc[i:i, "pctchg_pos"] = df_merged_prices.iloc[i]["sell_nominal_pos"] / df_merged_prices.iloc[i]["buy_nominal_pos"] - 1

    return df_merged_prices


def _build_perf_metrics(df_pnl: pd.DataFrame):

    consecutive_wins = qs.stats.consecutive_wins(returns=df_pnl["pctchg_pos"])
    consecutive_losses = qs.stats.consecutive_losses(returns=df_pnl["pctchg_pos"])
    nominal_profit = df_pnl["op_full_profit"].sum()

    df_metrics = pd.DataFrame({"nominal_profit": nominal_profit,
                            "consecutive_wins": consecutive_wins,
                            "consecutive_losses": consecutive_losses}, index=[0])

    return df_metrics
