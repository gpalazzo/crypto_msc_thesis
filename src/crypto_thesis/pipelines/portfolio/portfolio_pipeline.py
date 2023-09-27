# -*- coding: utf-8 -*-
"""Pipeline object having mainly func, inputs and outputs
`func` is the python function to be executed
`inputs` are either datasets or parameters defined in the conf/base directory
`outputs` are datasets defined in the catalog
- if the output is not defined in the catalog, then it becomes a MemoryDataSet
- MemoryDataSet persists as long as the Session is active
"""

from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.portfolio import (
    build_portfolio_metrics,
    build_portfolio_regr,
    build_regression_dataset,
)


def portfolio_pipeline() -> pipeline:

    _portfolio_pipeline = pipeline(
        Pipeline([
            # XGBoost
            node(func=build_portfolio_metrics,
                inputs=["xgboost_model_predict",
                        "window_nbr_lookup_test_multic",
                        "prm_binance",
                        "params:spine_preprocessing.target_name",
                        "params:portfolio_initial_money"],
                outputs=["xgboost_portfolio_pnl", "xgboost_portfolio_metrics"],
                name="run_xgboost_portfolio_metrics",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])

            # LSTM
            , node(func=build_portfolio_metrics,
                inputs=["lstm_model_predict",
                        "window_nbr_lookup_test_multic",
                        "prm_binance",
                        "params:spine_preprocessing.target_name",
                        "params:portfolio_initial_money"],
                outputs=["lstm_portfolio_pnl", "lstm_portfolio_metrics"],
                name="run_lstm_portfolio_metrics",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])

            # Logistic Regression
            , node(func=build_portfolio_metrics,
                inputs=["logreg_model_predict",
                        "window_nbr_lookup_test_nonmultic",
                        "prm_binance",
                        "params:spine_preprocessing.target_name",
                        "params:portfolio_initial_money"],
                outputs=["logreg_portfolio_pnl", "logreg_portfolio_metrics"],
                name="run_logreg_portfolio_metrics",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])
        ],
        tags=["portfolio_pipeline"]))

    _portfolios_regr_pipeline = pipeline(
        Pipeline([
            # build base dataset
            node(func=build_regression_dataset,
                inputs=["benchmark_trendfollowing_pnl",
                        "xgboost_portfolio_pnl",
                        "lstm_portfolio_pnl",
                        "logreg_portfolio_pnl"],
                outputs="portfolio_regression_data",
                name="run_build_regression_dataset",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"]),
            # calculate regressions
            # the order of input is extremely important to map the output
            node(func=build_portfolio_regr,
                inputs="portfolio_regression_data",
                outputs=["portfolio_regression_xgboost",
                         "portfolio_regression_lstm",
                         "portfolio_regression_logreg"],
                name="run_build_portfolio_regr",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])
        ],
        tags=["portfolios_regr_pipeline"]))

    return _portfolio_pipeline + _portfolios_regr_pipeline


def portfolio_pipeline_oos() -> pipeline:

    _portfolio_pipeline_oos = pipeline(
        Pipeline([
            # XGBoost
            node(func=build_portfolio_metrics,
                inputs=["xgboost_model_predict_oos",
                        "window_nbr_lookup_multic_oos",
                        "prm_binance_oos",
                        "params:spine_preprocessing.target_name",
                        "params:portfolio_initial_money"],
                outputs=["xgboost_portfolio_pnl_oos", "xgboost_portfolio_metrics_oos"],
                name="run_xgboost_portfolio_metrics_oos",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])

            # LSTM
            , node(func=build_portfolio_metrics,
                inputs=["lstm_model_predict_oos",
                        "window_nbr_lookup_multic_oos",
                        "prm_binance_oos",
                        "params:spine_preprocessing.target_name",
                        "params:portfolio_initial_money"],
                outputs=["lstm_portfolio_pnl_oos", "lstm_portfolio_metrics_oos"],
                name="run_lstm_portfolio_metrics_oos",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])

            # Logistic Regression
            , node(func=build_portfolio_metrics,
                inputs=["logreg_model_predict_oos",
                        "window_nbr_lookup_multic_oos",
                        "prm_binance_oos",
                        "params:spine_preprocessing.target_name",
                        "params:portfolio_initial_money"],
                outputs=["logreg_portfolio_pnl_oos", "logreg_portfolio_metrics_oos"],
                name="run_logreg_portfolio_metrics_oos",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])
        ],
        tags=["portfolio_pipeline_oos"]))

    return _portfolio_pipeline_oos
