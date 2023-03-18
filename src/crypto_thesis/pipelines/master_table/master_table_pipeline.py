# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.master_table import build_master_table


def master_table_pipeline():

    _master_table_pipeline = pipeline(
        Pipeline([
            node(func=build_master_table,
                inputs=["fte_binance_multic",
                        "spine_labeled"],
                outputs=["master_table_multic", "window_nbr_lookup_multic"],
                name="run_master_table_multic",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"]),

            node(func=build_master_table,
                inputs=["fte_binance_nonmultic",
                        "spine_labeled"],
                outputs=["master_table_nonmultic", "window_nbr_lookup_nonmultic"],
                name="run_master_table_nonmultic",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])
        ],
        tags=["master_table_pipeline"]))

    return _master_table_pipeline
