# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline
from crypto_thesis.data_domains.benchmark import buy_and_hold_strategy, trend_following_strategy, pc1_index_strategy
from crypto_thesis.data_domains.portfolio import build_portfolio_metrics


def benchmark_pipeline():

    _benchmark_strategies = pipeline(
        Pipeline([
            node(func=buy_and_hold_strategy,
                inputs=["window_nbr_lookup_multic",
                        "prm_binance",
                        "params:spine_preprocessing.target_name"],
                outputs="benchmark_buyhold_strat",
                name="run_benchmark_buyhold_strat",
                tags=["all_except_raw", "all_except_raw_prm"]),

            node(func=trend_following_strategy,
                inputs=["spine_preprocessing",
                        "window_nbr_lookup_multic"],
                outputs="benchmark_trendfollowing_strat",
                name="run_benchmark_trendfollowing_strat",
                tags=["all_except_raw", "all_except_raw_prm"]),

            # ******* WORK IN PROGRESS *******
            # node(func=pc1_index_strategy,
            #     inputs=["window_nbr_lookup_multic",
            #             "prm_binance"],
            #     outputs="benchmark_pc1_index_strat",
            #     name="run_benchmark_pc1_index_strat",
            #     tags=["all_except_raw", "all_except_raw_prm"])
            ],
        tags=["benchmark_strategies_pipeline"]))

    _benchmark_metrics = pipeline(
        # buy and hold strategy doesn't have calculated metrics here, only in the notebook
        Pipeline([
            node(func=build_portfolio_metrics,
                inputs=["benchmark_trendfollowing_strat",
                        "window_nbr_lookup_multic",
                        "prm_binance",
                        "params:spine_preprocessing.target_name",
                        "params:portfolio_initial_money"],
                outputs=["benchmark_trendfollowing_pnl", "benchmark_trendfollowing_metrics"],
                name="run_benchmark_trendfollowing_portfolio",
                tags=["all_except_raw", "all_except_raw_prm"])
            ],
        tags=["benchmark_metrics_pipeline"]))

    return _benchmark_strategies + _benchmark_metrics
