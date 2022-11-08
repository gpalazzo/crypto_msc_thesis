# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.spine import (
    spine_build_target_labels,
    spine_preprocessing,
)


def spine_pipeline():

    _spine_pipeline = pipeline(
        Pipeline([
            node(func=spine_preprocessing,
                inputs=["prm_binance",
                        "params:spine_preprocessing"],
                outputs=["spine_preprocessing", "spine_log_ret"],
                name="run_spine_preprocessing"),

            node(func=spine_build_target_labels,
                inputs=["spine_preprocessing", "spine_log_ret", "params:spine_labeling"],
                outputs="spine_labeled",
                name="run_spine_label")
        ],
        tags=["spine_pipeline"]))

    return _spine_pipeline