# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.benchmark import build_benchmark_metrics


def benchmark_pipeline():

    _benchmark_pipeline = pipeline(
        Pipeline([
            node(func=build_benchmark_metrics,
                inputs=["prm_manual_input",
                        "xgboost_model_predict",
                        "window_nbr_lookup",
                        "params:portfolio_initial_money"],
                outputs=["benchmark_pnl", "benchmark_metrics"],
                name="run_benchmark_metrics",
                tags=["all_except_raw", "all_except_raw_prm"]),
            ],
        tags=["benchmark_pipeline"]))

    return _benchmark_pipeline
