# -*- coding: utf-8 -*-
import pandas as pd

from crypto_thesis.utils import build_log_return


def manual_input_prm(df: pd.DataFrame,
                    start_date: str,
                    end_date: str) -> pd.DataFrame:

    df = df[df["date"].between(start_date, end_date)]
    df.loc[:, "pctchg"] = df["close_px"].pct_change()
    df = build_log_return(df=df, ref_col="close_px")

    df = df.drop(columns=["shift"])

    return df
