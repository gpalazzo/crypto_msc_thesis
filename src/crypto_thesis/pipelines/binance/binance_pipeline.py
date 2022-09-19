# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.binance import binance_prm, binance_raw


def binance_pipeline():

    raw_binance_pipeline = pipeline(
        Pipeline([node(func=binance_raw,
            inputs=["params:raw_binance_get_data"],
            outputs="raw_binance",
            name="run_binance_raw")],
        tags=["binance_pipeline"]))

    prm_binance_pipeline = pipeline(
        Pipeline([node(func=binance_prm,
            inputs=["raw_binance"],
            outputs="prm_binance",
            name="run_binance_prm")],
        tags=["binance_pipeline"]))

    return raw_binance_pipeline + prm_binance_pipeline
