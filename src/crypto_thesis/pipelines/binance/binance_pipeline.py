# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.binance import binance_fte, binance_prm, binance_raw


def binance_pipeline():

    _binance_pipeline = pipeline(
        Pipeline([
            node(func=binance_raw,
                inputs="params:raw_binance_get_data",
                outputs="raw_binance",
                name="run_binance_raw"),

            node(func=binance_prm,
                inputs=["raw_binance",
                        "params:min_years_existence",
                        "params:raw_binance_get_data.end_date"],
                outputs="prm_binance",
                name="run_binance_prm"),

            node(func=binance_fte,
                inputs=["prm_binance", "spine_preprocessing", "params:spine_preprocessing"],
                outputs="fte_binance",
                name="run_binance_fte")
            ],
        tags=["binance_pipeline"]))

    return _binance_pipeline
