# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.yahoo_finance import (
    yahoo_finance_prm,
    yahoo_finance_raw,
)


def yahoo_finance_pipeline():

    raw_yahoo_finance_pipeline = pipeline(
        Pipeline([node(func=yahoo_finance_raw,
            inputs=["params:raw_yahoo_finance_get_data"],
            outputs="raw_yahoo_finance",
            name="run_yahoo_finance_raw")],
        tags=["run_yahoo_finance_pipeline"]))

    prm_yahoo_finance_pipeline = pipeline(
        Pipeline([node(func=yahoo_finance_prm,
            inputs=["raw_yahoo_finance",
                    "params:raw_yahoo_finance_get_data"],
            outputs="prm_yahoo_finance",
            name="run_yahoo_finance_prm")],
        tags=["run_yahoo_finance_pipeline"]))

    return raw_yahoo_finance_pipeline + prm_yahoo_finance_pipeline
