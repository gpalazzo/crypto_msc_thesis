# -*- coding: utf-8 -*-
import logging
import math
from typing import Union

import numpy as np
import pandas as pd
import quantstats as qs

logger = logging.getLogger(__name__)


def build_portfolio_metrics(df_predict: pd.DataFrame,
                window_lookup: pd.DataFrame,
                prices_df: pd.DataFrame,
                target_name: str,
                portfolio_initial_money: Union[int, float]
                ) -> pd.DataFrame:

    window_lookup = window_lookup.set_index("window_nbr")
    df = df_predict.merge(window_lookup, left_index=True, right_index=True, how="inner")
    df = df.sort_index()

    assert df.shape[0] == df_predict.shape[0]

    # buy actions will only be taken when y_pred = 1 by construction
    # y_pred = 1 means the price level will be top compared to current market state
    df_top = df[df["y_pred"] == 1]

    df_pnl = _build_portfolio_pnl(df_top=df_top,
                                prices_df=prices_df,
                                target_name=target_name,
                                portfolio_initial_money=portfolio_initial_money)

    df_perf_metr = _build_perf_metrics(df_top=df_top,
                                    df_pnl=df_pnl,
                                    prices_df=prices_df,
                                    target_name=target_name)

    return df_pnl, df_perf_metr


def _build_portfolio_pnl(df_top: pd.DataFrame,
                            prices_df: pd.DataFrame,
                            target_name: str,
                            portfolio_initial_money: Union[int, float]) -> pd.DataFrame:

    df_target_prices = prices_df[prices_df["symbol"] == target_name] \
        [["open_time", "close"]].rename(columns={"open_time": "close_time"})

    # get close time price
    df_merged_prices = df_top.merge(df_target_prices,
                                on="close_time",
                                how="inner") \
                            .rename(columns={"close": "close_time_price"})
    assert df_top.shape[0] == df_merged_prices.shape[0], "Data loss when joining predictions with prices"

    # get target time price
    df_target_prices = df_target_prices.rename(columns={"close_time": "close_time_px"})
    df_merged_prices = df_merged_prices.merge(df_target_prices,
                                            left_on="target_time",
                                            right_on="close_time_px",
                                            how="inner") \
                            .rename(columns={"close": "target_time_price"}) \
                            .drop(columns=["close_time_px"])

    df_merged_prices.loc[:, "op_unit_profit"] = df_merged_prices["target_time_price"] - \
                                                 df_merged_prices["close_time_price"]

    df_merged_prices = df_merged_prices.sort_values(by="close_time").reset_index(drop=True)

    for i, row in df_merged_prices.iterrows():
        if i == 0: #first trade
            df_merged_prices.loc[i:i, "stock_qty"] = _round_decimals_down(portfolio_initial_money / row.close_time_price)
            df_merged_prices.loc[i:i, "op_full_profit"] = df_merged_prices.iloc[i]["stock_qty"] * df_merged_prices.iloc[i]["op_unit_profit"]
            df_merged_prices.loc[i:i, "buy_nominal_pos"] = df_merged_prices.iloc[i]["stock_qty"] * df_merged_prices.iloc[i]["close_time_price"]
            df_merged_prices.loc[i:i, "sell_nominal_pos"] = df_merged_prices.iloc[i]["stock_qty"] * df_merged_prices.iloc[i]["target_time_price"]
            df_merged_prices.loc[i:i, "residual_value"] = portfolio_initial_money - df_merged_prices.iloc[i]["buy_nominal_pos"]
        else:
            _total_curr_money = df_merged_prices.iloc[i-1].sell_nominal_pos + df_merged_prices.iloc[i-1].residual_value
            df_merged_prices.loc[i:i, "stock_qty"] = _round_decimals_down(_total_curr_money / row.close_time_price)
            df_merged_prices.loc[i:i, "op_full_profit"] = df_merged_prices.iloc[i]["stock_qty"] * df_merged_prices.iloc[i]["op_unit_profit"]
            df_merged_prices.loc[i:i, "buy_nominal_pos"] = df_merged_prices.iloc[i]["stock_qty"] * df_merged_prices.iloc[i]["close_time_price"]
            df_merged_prices.loc[i:i, "sell_nominal_pos"] = df_merged_prices.iloc[i]["stock_qty"] * df_merged_prices.iloc[i]["target_time_price"]
            df_merged_prices.loc[i:i, "residual_value"] = _total_curr_money - df_merged_prices.iloc[i]["buy_nominal_pos"]

    df_merged_prices.loc[:, "pctchg_sell_pos"] = df_merged_prices["sell_nominal_pos"].pct_change().fillna(0)

    return df_merged_prices


def _build_perf_metrics(df_top: pd.DataFrame,
                        df_pnl: pd.DataFrame,
                        prices_df: pd.DataFrame,
                        target_name: str):

    final_df = pd.DataFrame()

    df_target_prices = prices_df[prices_df["symbol"] == target_name] \
        [["open_time", "close"]].rename(columns={"open_time": "close_time"})
    df_target_prices.loc[:, "pct_chg"] = df_target_prices["close"].pct_change().fillna(0)

    for start, end in zip(df_top["open_time"], df_top["close_time"]):
        df_px_aux = df_target_prices[df_target_prices["close_time"].between(start, end)]
        final_df = pd.concat([final_df, df_px_aux])

    # periods = 360 because crypto trades 24/7
    sharpe = qs.stats.sharpe(returns=df_pnl["pctchg_sell_pos"], periods=360, annualize=True)
    profit_factor = qs.stats.profit_factor(returns=df_pnl["pctchg_sell_pos"]) #profit factor = win/loss
    sortino = qs.stats.sortino(returns=df_pnl["pctchg_sell_pos"], periods=360, annualize=True)
    consecutive_wins = qs.stats.consecutive_wins(returns=df_pnl["pctchg_sell_pos"])
    consecutive_losses = qs.stats.consecutive_losses(returns=df_pnl["pctchg_sell_pos"])

    max_drawdown = _get_max_drawdown(df=df_pnl)

    df_metrics = pd.DataFrame({"annual_sharpe": sharpe,
                            "profit_factor_pct": profit_factor,
                            "annual_sortino": sortino,
                            "max_drawdown_pct": max_drawdown,
                            "consecutive_wins": consecutive_wins,
                            "consecutive_losses": consecutive_losses}, index=[0])

    return df_metrics


def _get_max_drawdown(df: pd.DataFrame) -> float:
    df.loc[:, "logret_pos"] = np.log(1 + df["pctchg_sell_pos"])
    min_accum_value = df["logret_pos"].min()
    return np.exp(min_accum_value) - 1


def _round_decimals_down(number: float, decimals: int = 4):
    """
    Returns a value rounded down to a specific number of decimal places.
    """

    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals

    return math.floor(number * factor) / factor
