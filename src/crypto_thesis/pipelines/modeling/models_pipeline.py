# -*- coding: utf-8 -*-
"""Pipeline object having mainly func, inputs and outputs
`func` is the python function to be executed
`inputs` are either datasets or parameters defined in the conf/base directory
`outputs` are datasets defined in the catalog
- if the output is not defined in the catalog, then it becomes a MemoryDataSet
- MemoryDataSet persists as long as the Session is active
"""

from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.modeling import (
    logreg_model_fit,
    logreg_model_predict,
    logreg_model_reporting,
    lstm_model_fit,
    lstm_model_predict,
    lstm_model_reporting,
    xgboost_model_fit,
    xgboost_model_predict,
    xgboost_model_reporting,
)


def ml_models_pipeline() -> pipeline:

    xgboost_pipeline = pipeline(
        Pipeline([
            # model fit
            node(func=xgboost_model_fit,
                inputs=["master_table_multic",
                        "params:train_test_cutoff_date",
                        "params:xgboost_model_params",
                        "params:xgboost_optimize_params",
                        "params:xgboost_default_params"],
                outputs=["xgboost_fitted_model", "xgboost_optimized_params",
                        "xgboost_features_train", "xgboost_target_train",
                        "xgboost_features_test", "xgboost_target_test"],
                name="run_xgboost_fitting",
                tags=["all_except_raw", "all_except_binance"])
            
            # model predict
            , node(func=xgboost_model_predict,
                inputs=["xgboost_fitted_model", "xgboost_features_test"],
                outputs="xgboost_model_predict",
                name="run_xgboost_predicting",
                tags=["all_except_raw", "all_except_binance"])

            # model evaluate
            , node(func=xgboost_model_reporting,
                inputs=["xgboost_fitted_model",
                        "xgboost_features_test",
                        "xgboost_target_test",
                        "xgboost_model_predict",
                        "master_table_multic",
                        "params:model_data_interval",
                        "params:spine_preprocessing",
                        "params:spine_labeling",
                        "params:train_test_cutoff_date",
                        "params:slct_topN_features",
                        "params:min_years_existence"],
                outputs="xgboost_model_reporting",
                name="run_xgboost_reporting",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])
        ],
        tags=["xgboost_pipeline"]))

    lstm_pipeline = pipeline(
        Pipeline([
            # model fit
            node(func=lstm_model_fit,
                inputs=["master_table_multic",
                        "params:train_test_cutoff_date",
                        "params:lstm_timestamp_seq_length"],
                outputs=["lstm_fitted_model", "lstm_epoch_train_history",
                        "lstm_features_train", "lstm_target_train",
                        "lstm_features_test", "lstm_target_test"],
                name="run_lstm_fitting",
                tags=["all_except_raw", "all_except_binance"])

            # model predict
            , node(func=lstm_model_predict,
                inputs=["lstm_fitted_model",
                        "lstm_features_test",
                        "lstm_target_test",
                        "params:lstm_timestamp_seq_length"],
                outputs="lstm_model_predict",
                name="run_lstm_predicting",
                tags=["all_except_raw", "all_except_binance"])

            # model evaluate
            , node(func=lstm_model_reporting,
                inputs=["lstm_fitted_model",
                        "lstm_features_test",
                        "lstm_target_test",
                        "lstm_model_predict",
                        "master_table_multic",
                        "params:model_data_interval",
                        "params:spine_preprocessing",
                        "params:spine_labeling",
                        "params:train_test_cutoff_date",
                        "params:slct_topN_features",
                        "params:min_years_existence"],
                outputs="lstm_model_reporting",
                name="run_lstm_reporting",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])
        ],
        tags=["lstm_pipeline"]))

    logreg_pipeline = pipeline(
        Pipeline([
            # model fit
            node(func=logreg_model_fit,
                inputs=["master_table_nonmultic",
                        "params:train_test_cutoff_date",
                        "params:logreg_model_params",
                        "params:logreg_optimize_params",
                        "params:logreg_default_params"],
                outputs=["logreg_fitted_model", "logreg_optimized_params",
                        "logreg_features_train", "logreg_target_train",
                        "logreg_features_test", "logreg_target_test"],
                name="run_logreg_fitting",
                tags=["all_except_raw", "all_except_binance"])

            # model predict
            , node(func=logreg_model_predict,
                inputs=["logreg_fitted_model",
                        "logreg_features_test"],
                outputs="logreg_model_predict",
                name="run_logreg_predicting",
                tags=["all_except_raw", "all_except_binance"])

            # model evaluate
            , node(func=logreg_model_reporting,
                inputs=["logreg_fitted_model",
                        "logreg_features_test",
                        "logreg_target_test",
                        "logreg_model_predict",
                        "master_table_nonmultic",
                        "params:model_data_interval",
                        "params:spine_preprocessing",
                        "params:spine_labeling",
                        "params:train_test_cutoff_date",
                        "params:slct_topN_features",
                        "params:min_years_existence"],
                outputs="logreg_model_reporting",
                name="run_logreg_reporting",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])
        ],
        tags=["logreg_pipeline"]))

    return xgboost_pipeline + lstm_pipeline + logreg_pipeline
