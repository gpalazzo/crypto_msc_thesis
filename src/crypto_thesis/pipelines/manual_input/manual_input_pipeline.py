# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.manual_input import manual_input_prm


def manual_input_pipeline():

    _manual_input_pipeline = pipeline(
        Pipeline([
            node(func=manual_input_prm,
                inputs=["raw_manual_input",
                        "params:mi_start_date",
                        "params:mi_end_date"],
                outputs="prm_manual_input",
                name="run_manual_input_prm",
                tags=["all_except_raw"]),
            ],
        tags=["manual_input_pipeline"]))

    return _manual_input_pipeline
