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
                outputs=["portfolio_pnl", "portfolio_metrics"],
                name="run_portfolio_metrics",
                tags=["all_except_raw", "all_except_binance"])
        ],
        tags=["portfolio_pipeline"]))

    return _portfolio_pipeline
