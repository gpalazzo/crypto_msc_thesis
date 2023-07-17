# -*- coding: utf-8 -*-
"""Pipeline object having mainly func, inputs and outputs
`func` is the python function to be executed
`inputs` are either datasets or parameters defined in the conf/base directory
`outputs` are datasets defined in the catalog
- if the output is not defined in the catalog, then it becomes a MemoryDataSet
- MemoryDataSet persists as long as the Session is active
"""

from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.portfolio import build_portfolio_metrics


def portfolio_pipeline() -> pipeline:

    _portfolio_pipeline = pipeline(
        Pipeline([
            # XGBoost
            node(func=build_portfolio_metrics,
                inputs=["xgboost_model_predict",
                        "window_nbr_lookup_multic",
                        "prm_binance",
                        "params:spine_preprocessing.target_name",
                        "params:portfolio_initial_money"],
                outputs=["xgboost_portfolio_pnl", "xgboost_portfolio_metrics"],
                name="run_xgboost_portfolio_metrics",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])

            # LSTM
            , node(func=build_portfolio_metrics,
                inputs=["lstm_model_predict",
                        "window_nbr_lookup_multic",
                        "prm_binance",
                        "params:spine_preprocessing.target_name",
                        "params:portfolio_initial_money"],
                outputs=["lstm_portfolio_pnl", "lstm_portfolio_metrics"],
                name="run_lstm_portfolio_metrics",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])

            # Logistic Regression
            , node(func=build_portfolio_metrics,
                inputs=["logreg_model_predict",
                        "window_nbr_lookup_nonmultic",
                        "prm_binance",
                        "params:spine_preprocessing.target_name",
                        "params:portfolio_initial_money"],
                outputs=["logreg_portfolio_pnl", "logreg_portfolio_metrics"],
                name="run_logreg_portfolio_metrics",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])
        ],
        tags=["portfolio_pipeline"]))

    return _portfolio_pipeline
