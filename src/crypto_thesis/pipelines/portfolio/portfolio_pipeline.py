# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.portfolio import build_portfolio_metrics


def portfolio_pipeline():

    _portfolio_pipeline = pipeline(
        Pipeline([
            node(func=build_portfolio_metrics,
                inputs=["xgboost_model_predict",
                        "window_nbr_lookup",
                        "prm_binance",
                        "params:spine_preprocessing.target_name",
                        "params:portfolio_initial_money"],
                outputs=["xgboost_portfolio_pnl", "xgboost_portfolio_metrics"],
                name="run_xgboost_portfolio_metrics",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])

            , node(func=build_portfolio_metrics,
                inputs=["lstm_model_predict",
                        "window_nbr_lookup",
                        "prm_binance",
                        "params:spine_preprocessing.target_name",
                        "params:portfolio_initial_money"],
                outputs=["lstm_portfolio_pnl", "lstm_portfolio_metrics"],
                name="run_lstm_portfolio_metrics",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])
        ],
        tags=["portfolio_pipeline"]))

    return _portfolio_pipeline
