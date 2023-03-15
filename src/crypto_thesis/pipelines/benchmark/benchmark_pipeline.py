# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.benchmark import build_benchmark_metrics


def benchmark_pipeline():

    _benchmark_pipeline = pipeline(
        Pipeline([
            node(func=build_benchmark_metrics,
                inputs=["prm_manual_input",
                        "xgboost_model_predict",
                        "window_nbr_lookup_multic",
                        "params:portfolio_initial_money"],
                outputs=["benchmark_xgboost_pnl", "benchmark_xgboost_metrics"],
                name="run_benchmark_xgboost_metrics",
                tags=["all_except_raw", "all_except_raw_prm"]),

            node(func=build_benchmark_metrics,
                inputs=["prm_manual_input",
                        "lstm_model_predict",
                        "window_nbr_lookup_multic",
                        "params:portfolio_initial_money"],
                outputs=["benchmark_lstm_pnl", "benchmark_lstm_metrics"],
                name="run_benchmark_lstm_metrics",
                tags=["all_except_raw", "all_except_raw_prm"]),

            node(func=build_benchmark_metrics,
                inputs=["prm_manual_input",
                        "logreg_model_predict",
                        "window_nbr_lookup_multic",
                        "params:portfolio_initial_money"],
                outputs=["benchmark_logreg_pnl", "benchmark_logreg_metrics"],
                name="run_benchmark_logreg_metrics",
                tags=["all_except_raw", "all_except_raw_prm"])
            ],
        tags=["benchmark_pipeline"]))

    return _benchmark_pipeline
