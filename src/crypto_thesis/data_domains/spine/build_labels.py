# -*- coding: utf-8 -*-
from typing import Dict

import pandas as pd


def spine_build_target_labels(df: pd.DataFrame, df_log_ret: pd.DataFrame, label_params: Dict[str, float]) -> pd.DataFrame:

    std_df = _build_stdev_by_window(df=df, df_log_ret=df_log_ret)
    final_df = df.merge(std_df, on=["open_time", "close_time"], how="inner")

    assert df.shape[0] == final_df.shape[0], "Data loss, please review."

    final_df.loc[:, "label"] = final_df.apply(lambda col: "top" if col["log_return"] >= col["std"] * label_params["tau"] else "bottom", axis=1)

    return final_df


def _build_stdev_by_window(df: pd.DataFrame, df_log_ret: pd.DataFrame) -> pd.DataFrame:

    std_df = pd.DataFrame()

    for start, end in zip(df["open_time"], df["close_time"]):
        df_filter = df_log_ret[df_log_ret["close_time"].between(start, end)]

        _std = df_filter["log_return"].std(ddof=1) #sampled stdev
        df_aux = pd.DataFrame({"open_time": start, "close_time": end, "std": _std}, index=[0])

        std_df = pd.concat([std_df, df_aux])

    return std_df
