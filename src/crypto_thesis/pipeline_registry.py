# -*- coding: utf-8 -*-
"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from crypto_thesis.pipelines.binance import binance_pipeline
from crypto_thesis.pipelines.spine import spine_pipeline
from crypto_thesis.pipelines.yahoo_finance import yahoo_finance_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {"__default__": pipeline([yahoo_finance_pipeline() +
                                    binance_pipeline() +
                                    spine_pipeline()])}
