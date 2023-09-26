# -*- coding: utf-8 -*-
"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from crypto_thesis.pipelines.benchmark import benchmark_pipeline
from crypto_thesis.pipelines.binance import binance_pipeline, binance_pipeline_oos
from crypto_thesis.pipelines.master_table import (
    master_table_pipeline,
    master_table_pipeline_oos,
)
from crypto_thesis.pipelines.modeling import ml_models_pipeline, ml_models_pipeline_oos
from crypto_thesis.pipelines.portfolio import portfolio_pipeline, portfolio_pipeline_oos
from crypto_thesis.pipelines.spine import spine_pipeline, spine_pipeline_oos


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {"__default__": pipeline([binance_pipeline() +
                                    binance_pipeline_oos() +
                                    spine_pipeline() +
                                    spine_pipeline_oos() +
                                    master_table_pipeline() +
                                    master_table_pipeline_oos() +
                                    benchmark_pipeline() +
                                    ml_models_pipeline() +
                                    ml_models_pipeline_oos() +
                                    portfolio_pipeline() +
                                    portfolio_pipeline_oos()])}
