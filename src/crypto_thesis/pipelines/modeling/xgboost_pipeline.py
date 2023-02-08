# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline


def xgboost_model_pipeline():

    _xgboost_pipeline = pipeline(
        Pipeline([
            node(func=xgboost_model,
                inputs="master_table",
                outputs="xgboost",
                name="run_xgboost")
        ],
        tags=["xgboost_pipeline"]))

    return _xgboost_pipeline
