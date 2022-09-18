# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.pipelines.yahoo_finance import yahoo_finance_raw


def yahoo_finance_pipeline():

    raw_yahoo_finance_pipeline = pipeline(Pipeline([node(func=yahoo_finance_raw, inputs=[], outputs=[], name="run_yahoo_finance_raw")]))
