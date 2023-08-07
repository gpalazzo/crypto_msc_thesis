# -*- coding: utf-8 -*-
from functools import reduce
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper


def build_regression_dataset(benchm_df: pd.DataFrame,
                               xgboost_df: pd.DataFrame,
                               lstm_df: pd.DataFrame,
                               logreg_df: pd.DataFrame) -> pd.DataFrame:
    """Build aggregated dataset with all portfolio's positions

    Args:
        benchm_df (pd.DataFrame): dataframe with benchmark's portfolio
        xgboost_df (pd.DataFrame): dataframe with xgboost's portfolio
        lstm_df (pd.DataFrame): dataframe with lstm's portfolio
        logreg_df (pd.DataFrame): dataframe with logistic regression's portfolio

    Returns:
        pd.DataFrame: dataframe with all portfolio's positions together
    """

    benchm_df.loc[:, "portfolio_ref"] = "benchmark"
    xgboost_df.loc[:, "portfolio_ref"] = "xgboost"
    lstm_df.loc[:, "portfolio_ref"] = "lstm"
    logreg_df.loc[:, "portfolio_ref"] = "logreg"

    dfs = _aggregate_dfs(benchm_df, xgboost_df, lstm_df, logreg_df)
    dfs = _pivot_portfolio(dfs)

    total_rows = sum([df.shape[0] for df in dfs])
    max_rows = max([df.shape[0] for df in dfs])

    df_all = reduce(lambda left, right: pd.merge(left, right, on=["date"], how="outer"), dfs)
    assert df_all.shape[0] >= max_rows and df_all.shape[0] <= total_rows, "Wrong number of lines after join, review"

    df_all = df_all.fillna(0)

    return df_all


def build_portfolio_regr(df: pd.DataFrame) -> List[RegressionResultsWrapper]:
    """Build portfolio regression of the models against the benchmark

    Args:
        df (pd.DataFrame): dataframe with portfolio data to apply regression

    Returns:
        List[RegressionResultsWrapper]: list of fitted models (in order as defined by variable `MODELS`)
    """

    MODELS = ["xgboost", "lstm", "logreg"]
    PAIRS = [["benchmark", model] for model in MODELS]
    CONSTANT_COL = "constant"

    df.loc[:, CONSTANT_COL] = 1
    models = []

    for pair in PAIRS:
        dfaux = df[pair + [CONSTANT_COL]]
        y = dfaux[pair[0]].copy()
        X = dfaux.drop(columns=[pair[0]]).copy()

        model = sm.OLS(y, X).fit()
        models.append(model)

    return models


def _aggregate_dfs(*args: Tuple[pd.DataFrame]) -> List[pd.DataFrame]:
    """Accumulate dataframes on a daily basis by summing the intraday positions change

    Args:
        args (Tuple[pd.DataFrame]): N-length tuple of dataframes to be aggregated

    Returns:
        List[pd.DataFrame]: list of dataframes aggregated on a dialy basis (same order as input)
    """

    dfs = []

    for _df in args:
        df = _df[["close_time", "portfolio_ref", "pctchg_pos"]]

        df.loc[:, "date"] = df["close_time"].dt.date
        df.loc[:, "logrets_pos"] = np.log(1 + df["pctchg_pos"])

        df = df.groupby(["date", "portfolio_ref"]) \
                ["logrets_pos"].sum() \
                .reset_index()

        df.loc[:, "pctchg_pos"] = np.exp(df["logrets_pos"]) - 1
        df = df.drop(columns=["logrets_pos"])
        dfs.append(df)

    return dfs


def _pivot_portfolio(args: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Pivot dataframe in a way each portfolio goes to a column

    Args:
        args (List[pd.DataFrame]): list of dataframes to be pivotted

    Returns:
        List[pd.DataFrame]: list of pivotted dataframe (same order as input)
    """

    dfs = []
    VALUES_COL = "pctchg_pos"

    for _df in args:
        df = _df.pivot(index="date", columns=["portfolio_ref"], values=[VALUES_COL])
        df = df[VALUES_COL].reset_index()
        dfs.append(df)

    return dfs
