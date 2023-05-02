# -*- coding: utf-8 -*-
import math
from typing import Union

import pandas as pd
import quantstats as qs


def build_benchmark_metrics(
                df_benchmark: pd.DataFrame,
                portfolio_initial_money: Union[int, float]) -> pd.DataFrame:

    df_pnl = _build_benchmark_pnl(prices_df=df_benchmark,
                                portfolio_initial_money=portfolio_initial_money)

    df_perf_metr = _build_perf_metrics(df_pnl=df_pnl)

    return df_pnl, df_perf_metr


def _build_benchmark_pnl(prices_df: pd.DataFrame,
                        portfolio_initial_money: Union[int, float]) -> pd.DataFrame:
    """Assumption: since we don't have intraday prices, let's get the dates with trades (label = 1)
    and collect the benchmark prices. Then let's assume each subsequential is a target time
    """

    prices_df = prices_df.drop(columns=["pctchg", "log_return"])
    prices_df = prices_df.rename(columns={"date": "close_time", "close_px": "close_time_price"})
    prices_df = prices_df.sort_values(by="close_time").reset_index(drop=True)

    prices_df.loc[:, "target_time"] = prices_df["close_time"].shift(-1)
    prices_df.loc[:, "target_time_price"] = prices_df["close_time_price"].shift(-1)
    prices_df = prices_df.dropna()

    prices_df.loc[:, "op_unit_profit"] = prices_df["target_time_price"] - \
                                                 prices_df["close_time_price"]

    for i, row in prices_df.iterrows():
        if i == 0: #first trade
            prices_df.loc[i:i, "stock_qty"] = math.floor(portfolio_initial_money / row.close_time_price)
            prices_df.loc[i:i, "op_full_profit"] = prices_df.iloc[i]["stock_qty"] * prices_df.iloc[i]["op_unit_profit"]
            prices_df.loc[i:i, "buy_nominal_pos"] = prices_df.iloc[i]["stock_qty"] * prices_df.iloc[i]["close_time_price"]
            prices_df.loc[i:i, "sell_nominal_pos"] = prices_df.iloc[i]["stock_qty"] * prices_df.iloc[i]["target_time_price"]
            prices_df.loc[i:i, "residual_value"] = portfolio_initial_money - prices_df.iloc[i]["buy_nominal_pos"]
            prices_df.loc[i:i, "pctchg_pos"] = prices_df.iloc[i]["sell_nominal_pos"] / prices_df.iloc[i]["buy_nominal_pos"] - 1
        else:
            _total_curr_money = prices_df.iloc[i-1].sell_nominal_pos + prices_df.iloc[i-1].residual_value
            prices_df.loc[i:i, "stock_qty"] = math.floor(_total_curr_money / row.close_time_price)
            prices_df.loc[i:i, "op_full_profit"] = prices_df.iloc[i]["stock_qty"] * prices_df.iloc[i]["op_unit_profit"]
            prices_df.loc[i:i, "buy_nominal_pos"] = prices_df.iloc[i]["stock_qty"] * prices_df.iloc[i]["close_time_price"]
            prices_df.loc[i:i, "sell_nominal_pos"] = prices_df.iloc[i]["stock_qty"] * prices_df.iloc[i]["target_time_price"]
            prices_df.loc[i:i, "residual_value"] = _total_curr_money - prices_df.iloc[i]["buy_nominal_pos"]
            prices_df.loc[i:i, "pctchg_pos"] = prices_df.iloc[i]["sell_nominal_pos"] / prices_df.iloc[i]["buy_nominal_pos"] - 1

    return prices_df


def _build_perf_metrics(df_pnl: pd.DataFrame):

    consecutive_wins = qs.stats.consecutive_wins(returns=df_pnl["pctchg_pos"])
    consecutive_losses = qs.stats.consecutive_losses(returns=df_pnl["pctchg_pos"])
    nominal_profit = df_pnl["op_full_profit"].sum()

    df_metrics = pd.DataFrame({"nominal_profit": nominal_profit,
                            "consecutive_wins": consecutive_wins,
                            "consecutive_losses": consecutive_losses}, index=[0])

    return df_metrics
