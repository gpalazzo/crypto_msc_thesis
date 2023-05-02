# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.benchmark import build_benchmark_metrics, buy_and_hold_strategy


def benchmark_pipeline():

    _benchmark_strategies = pipeline(
        Pipeline([
            node(func=buy_and_hold_strategy,
                inputs=["window_nbr_lookup_multic",
                        "prm_binance",
                        "params:spine_preprocessing.target_name"],
                outputs="benchmark_buyhold_strat",
                name="run_benchmark_buyhold_strat",
                tags=["all_except_raw", "all_except_raw_prm"])
            ],
        tags=["benchmark_strategies_pipeline"]))

    _benchmark_metrics = pipeline(
        Pipeline([
            node(func=build_benchmark_metrics,
                inputs=["benchmark_buyhold_strat",
                        "params:portfolio_initial_money"],
                outputs=["benchmark_buyhold_pnl", "benchmark_buyhold_metrics"],
                name="run_benchmark_buyhold_portfolio",
                tags=["all_except_raw", "all_except_raw_prm"])
            ],
        tags=["benchmark_metrics_pipeline"]))

    return _benchmark_strategies + _benchmark_metrics
