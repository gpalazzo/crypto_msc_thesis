# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.portfolio import build_portfolio_pnl


def portfolio_pipeline():

    _portfolio_pipeline = pipeline(
        Pipeline([
            node(func=build_portfolio_pnl,
                inputs=["xgboost_model_predict",
                        "window_nbr_lookup",
                        "prm_binance",
                        "params:spine_preprocessing.target_name"],
                outputs="portfolio_pnl",
                name="run_portfolio_pnl")

            # , node(func=build_portfolio_metrics,
            #     inputs=["portfolio_pnl",
            #             "prm_binance",
            #             "params:spine_preprocessing.target_name"],
            #     outputs="portfolio_pnl",
            #     name="run_portfolio_pnl")
        ],
        tags=["portfolio_pipeline"]))

    return _portfolio_pipeline
