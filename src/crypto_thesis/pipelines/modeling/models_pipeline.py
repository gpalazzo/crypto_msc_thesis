# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.modeling import xgboost_model


def ml_models_pipeline():

    xgboost_pipeline = pipeline(
        Pipeline([
            node(func=xgboost_model,
                inputs=["master_table", "params:bars_window_size"],
                outputs="xgboost",
                name="run_xgboost")
        ],
        tags=["xgboost_pipeline"]))

    return xgboost_pipeline
