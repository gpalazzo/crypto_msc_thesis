# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.master_table import build_master_table


def master_table_pipeline():

    _master_table_pipeline = pipeline(
        Pipeline([
            node(func=build_master_table,
                inputs=["fte_binance",
                        "spine_labeled"],
                outputs="master_table",
                name="run_master_table")
        ],
        tags=["master_table_pipeline"]))

    return _master_table_pipeline