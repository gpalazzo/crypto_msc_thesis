# -*- coding: utf-8 -*-
from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.spine import spine_preprocessing


def spine_pipeline():

    spine_pipeline = pipeline(
        Pipeline([node(func=spine_preprocessing,
            inputs=["prm_binance",
                    "params:spine_preprocessing"],
            outputs="spine_preprocessing",
            name="run_spine_preprocessing")],
        tags=["spine_pipeline"]))

    return spine_pipeline
