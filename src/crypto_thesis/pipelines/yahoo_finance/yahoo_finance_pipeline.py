# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.yahoo_finance import yahoo_finance_raw


def yahoo_finance_pipeline():

    raw_yahoo_finance_pipeline = pipeline(
        Pipeline([node(func=yahoo_finance_raw,
            inputs=["params:raw_yahoo_finance_get_data"],
            outputs="raw_yahoo_finance",
            name="run_yahoo_finance_raw")]))

    return raw_yahoo_finance_pipeline
