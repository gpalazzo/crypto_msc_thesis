# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.modeling import (
    xgboost_model_fit,
    xgboost_model_predict,
    xgboost_model_reporting,
)


def ml_models_pipeline():

    xgboost_pipeline = pipeline(
        Pipeline([
            node(func=xgboost_model_fit,
                inputs="master_table",
                outputs=["xgboost_fitted_model",
                        "xgboost_features_train", "xgboost_target_train",
                        "xgboost_features_test", "xgboost_target_test"],
                name="run_xgboost_fitting")

            , node(func=xgboost_model_predict,
                inputs=["xgboost_fitted_model", "xgboost_features_test"],
                outputs="xgboost_model_predict",
                name="run_xgboost_predicting")

            , node(func=xgboost_model_reporting,
                inputs=["xgboost_fitted_model",
                        "xgboost_features_test",
                        "xgboost_target_test",
                        "xgboost_model_predict",
                        "params:model_data_interval"],
                outputs="xgboost_model_reporting",
                name="run_xgboost_reporting")
        ],
        tags=["xgboost_pipeline"]))

    return xgboost_pipeline
