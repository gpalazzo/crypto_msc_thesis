# -*- coding: utf-8 -*-
import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def spine_build_target_labels(df: pd.DataFrame,
                            df_log_ret: pd.DataFrame,
                            label_params: Dict[str, float],
                            class_bounds: Dict[str, float]) -> pd.DataFrame:

    std_df = _build_stdev_by_window(df=df, df_log_ret=df_log_ret)
    final_df = df.merge(std_df, on=["open_time", "close_time"], how="inner")

    assert df.shape[0] == final_df.shape[0], "Data loss, please review."

    # it has nulls for rows were the volume is equal to the volume bar size
    # so there's only data point and therefore there's no stdev
    final_df = final_df.dropna()

    # build log return between close_time price and target_time price
    final_df.loc[:, "close_to_tgt_time_logret"] = np.log(final_df["target_time_close"] \
                                                            / final_df["close_time_close"])
    # transform log rets into percent change
    final_df[["pctchg_cumsum", "close_to_tgt_time_pctchg"]] = final_df[["logret_cumsum", "close_to_tgt_time_logret"]].applymap(lambda row: np.exp(row) - 1)

    # by construction, every time the `close_to_tgt_time_logret` is negative, it means the price in
    # target_time is lower than close_time, so it's automatically a bottom label
    final_df = final_df.reset_index(drop=True)
    final_df_neg = final_df[final_df["close_to_tgt_time_pctchg"] < 0.0]
    final_df_neg.loc[:, "label"] = "bottom"
    final_df_pos = final_df.drop(final_df_neg.index)

    final_df_pos.loc[:, "label"] = final_df_pos.apply(lambda col: "top" \
                                                if col["close_to_tgt_time_pctchg"] >= \
                                                    col["pctchg_cumsum"] + col["std"] * label_params["tau"] \
                                                else "bottom" \
                                            , axis=1)

    final_df = pd.concat([final_df_neg, final_df_pos])

    logger.info("Checking for class balance")
    final_df = _balance_classes(df=final_df, class_bounds=class_bounds)

    _check_spine_quality(df=final_df, class_bounds=class_bounds)

    return final_df


def _build_stdev_by_window(df: pd.DataFrame, df_log_ret: pd.DataFrame) -> pd.DataFrame:

    std_df = pd.DataFrame()

    for start, end in zip(df["open_time"], df["close_time"]):
        df_filter = df_log_ret[df_log_ret["close_time"].between(start, end)]

        _std = df_filter["pctchg"].std(ddof=1) #sampled stdev
        df_aux = pd.DataFrame({"open_time": start, "close_time": end, "std": _std}, index=[0])

        std_df = pd.concat([std_df, df_aux])

    return std_df


def _balance_classes(df: pd.DataFrame,
                    class_bounds: Dict[str, float]) -> pd.DataFrame:

    label0_count, label1_count = df.label.value_counts()
    label0_pct = label0_count / df.shape[0]

    # check if any label is outside the desired range
    # there's no need to check both labels, if 1 is outside the range, the other will also be
    if not pd.Series(label0_pct). \
            between(class_bounds["lower"],
                    class_bounds["upper"]) \
                    [0]: #pull index [0] always works because it's only 1 label

        df = df.reset_index(drop=True)
        df_top = df[df["label"] == "top"]
        df_bottom = df.drop(df_top.index)

        # find out which class in unbalanced
        if label0_count > label1_count:
            df_bottom = df_bottom.sample(n=label1_count)
        else:
            df_top = df_top.sample(n=label0_count)

    else:
        logger.info("Class are balanced, skipping balancing method")

    df = pd.concat([df_bottom, df_top])

    return df


def _check_spine_quality(df: pd.DataFrame,
                        class_bounds: Dict[str, float]) -> None:

    # check label unbalancing
    labels_pct = (df.label.value_counts() / df.shape[0]).values
    assert any([(label_pct >= class_bounds["lower"] and \
                    label_pct <= class_bounds["upper"]) \
                for label_pct in labels_pct]), "Unbalanced classes, review."

    # check nulls
    assert df.isnull().sum().sum() == 0, "Spine contains null, review."
