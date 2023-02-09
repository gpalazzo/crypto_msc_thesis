# -*- coding: utf-8 -*-
import pandas as pd


def xgboost_model(master_table: pd.DataFrame) -> pd.DataFrame:
    for start, end in zip(master_table["open_time"], master_table["close_time"]):
        df_filter = master_table[master_table["close_time"].between(start, end)]
        # breakpoint()
