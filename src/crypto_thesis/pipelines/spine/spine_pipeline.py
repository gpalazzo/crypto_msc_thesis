# -*- coding: utf-8 -*-
"""Pipeline object having mainly func, inputs and outputs
`func` is the python function to be executed
`inputs` are either datasets or parameters defined in the conf/base directory
`outputs` are datasets defined in the catalog
- if the output is not defined in the catalog, then it becomes a MemoryDataSet
- MemoryDataSet persists as long as the Session is active
"""

from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.spine import (
    spine_build_target_labels,
    spine_preprocessing,
)


def spine_pipeline() -> pipeline:

    _spine_pipeline = pipeline(
        Pipeline([
            # prepare the dataset
            node(func=spine_preprocessing,
                inputs=["prm_binance",
                        "params:spine_preprocessing"],
                outputs=["spine_preprocessing", "spine_log_ret"],
                name="run_spine_preprocessing",
                tags=["all_except_raw", "all_except_binance"]),

            # generate the labels
            node(func=spine_build_target_labels,
                inputs=["spine_preprocessing",
                        "spine_log_ret",
                        "params:spine_labeling"],
                outputs="spine_labeled",
                name="run_spine_label",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])
        ],
        tags=["spine_pipeline"]))

    return _spine_pipeline
