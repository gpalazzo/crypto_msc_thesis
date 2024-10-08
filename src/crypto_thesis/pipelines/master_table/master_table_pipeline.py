# -*- coding: utf-8 -*-
"""Pipeline object having mainly func, inputs and outputs
`func` is the python function to be executed
`inputs` are either datasets or parameters defined in the conf/base directory
`outputs` are datasets defined in the catalog
- if the output is not defined in the catalog, then it becomes a MemoryDataSet
- MemoryDataSet persists as long as the Session is active
"""

from kedro.pipeline import Pipeline, node, pipeline

from crypto_thesis.data_domains.master_table import (
    build_master_table,
    build_master_table_oos,
)


def master_table_pipeline() -> pipeline:

    _master_table_pipeline = pipeline(
        Pipeline([
            # build master table with multicollinear features
            node(func=build_master_table,
                inputs=["fte_binance_multic",
                        "spine_labeled",
                        "params:mt_class_bounds",
                        "params:train_test_cutoff_date"],
                outputs=["master_table_train_multic", "window_nbr_lookup_train_multic",
                        "master_table_test_multic", "window_nbr_lookup_test_multic"],
                name="run_master_table_multic",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"]),

            # build master table without multicollinear features
            node(func=build_master_table,
                inputs=["fte_binance_nonmultic",
                        "spine_labeled",
                        "params:mt_class_bounds",
                        "params:train_test_cutoff_date"],
                outputs=["master_table_train_nonmultic", "window_nbr_lookup_train_nonmultic",
                        "master_table_test_nonmultic", "window_nbr_lookup_test_nonmultic"],
                name="run_master_table_nonmultic",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"])
        ],
        tags=["master_table_pipeline"]))

    return _master_table_pipeline


def master_table_pipeline_oos() -> pipeline:

    _master_table_pipeline_oos = pipeline(
        Pipeline([
            # build master table with multicollinear features
            node(func=build_master_table_oos,
                inputs=["fte_binance_multic_oos",
                        "spine_labeled_oos"],
                outputs=["master_table_multic_oos", "window_nbr_lookup_multic_oos"],
                name="run_master_table_multic_oos",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"]),

            # build master table without multicollinear features
            node(func=build_master_table_oos,
                inputs=["fte_binance_nonmultic_oos",
                        "spine_labeled_oos"],
                outputs=["master_table_nonmultic_oos", "window_nbr_lookup_nonmultic_oos"],
                name="run_master_table_nonmultic_oos",
                tags=["all_except_raw", "all_except_binance", "all_except_raw_prm"]),
        ],
        tags=["master_table_pipeline_oos"]))

    return _master_table_pipeline_oos
