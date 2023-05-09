# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.binance import binance_fte, binance_prm, binance_raw
from crypto_thesis.utils import apply_mic_fte_slct, apply_vif_fte_slct


def binance_pipeline():

    _binance_pipeline = pipeline(
        Pipeline([
            node(func=binance_raw,
                inputs="params:raw_binance_get_data",
                outputs="raw_binance",
                name="run_binance_raw"),

            node(func=binance_prm,
                inputs=["raw_binance",
                        "params:min_years_existence",
                        "params:raw_binance_get_data.end_date"],
                outputs="prm_binance",
                name="run_binance_prm",
                tags=["all_except_raw"]),

            node(func=binance_fte,
                inputs=["prm_binance",
                        "spine_labeled_balanced",
                        "params:spine_preprocessing"],
                outputs="fte_binance",
                name="run_binance_fte",
                tags=["all_except_raw", "all_except_raw_prm"]),

            node(func=apply_mic_fte_slct,
                inputs=["fte_binance",
                        "spine_labeled_balanced",
                        "params:train_test_cutoff_date",
                        "params:slct_topN_features"],
                outputs=["fte_binance_multic", "all_fte_multic_mic_binance"],
                name="run_binance_fte_multic",
                tags=["all_except_raw", "all_except_raw_prm"]),

            node(func=apply_vif_fte_slct,
                inputs=["fte_binance",
                        "spine_labeled_balanced",
                        "params:train_test_cutoff_date",
                        "params:slct_topN_features",
                        "params:vif_threshold"],
                outputs=["fte_binance_nonmultic", "all_fte_vif_binance", "all_fte_nonmultic_mic_binance"],
                name="run_binance_fte_nonmultic",
                tags=["all_except_raw", "all_except_raw_prm"])
            ],
        tags=["binance_pipeline"]))

    return _binance_pipeline
