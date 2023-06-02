# -*- coding: utf-8 -*-
from typing import Tuple, Union

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

INDEX_COL = ["open_time", "close_time"]


def apply_mic_fte_slct(df_ftes: pd.DataFrame,
                            spine_labeled: pd.DataFrame,
                            train_test_cutoff_date: str,
                            topN_features: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select feature using the Mutual Information Critetion (MIC) method

    Args:
        df_ftes (pd.DataFrame): dataframe containing the features
        spine_labeled (pd.DataFrame): spine dataset already with class labels
        train_test_cutoff_date (str): cutoff date to split dataset into train and test
        topN_features (int): amount of features to keep

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: dataframe with remaining features and 
        dataframe with MIC features' importance value
    """

    df_ftes_aux = df_ftes.copy() # do not change original data, it will be reused

    TARGET_COL = "label"
    df_ftes_aux = df_ftes_aux.dropna()

    spine_labeled = spine_labeled[["open_time", "close_time", TARGET_COL]]

    df = df_ftes_aux.merge(spine_labeled, on=["open_time", "close_time"], how="inner")
    assert df.shape[0] == df_ftes_aux.shape[0], "Data loss joining spine and ftes for feature selection, review."

    df = df[df["close_time"] < train_test_cutoff_date]
    df = df.set_index(INDEX_COL)
    df.loc[:, TARGET_COL] = df[TARGET_COL].replace({"top": 1, "bottom": 0})

    X_train_ftes = df.drop(columns=[TARGET_COL])
    y_train_ftes = df[[TARGET_COL]]

    selector = SelectKBest(mutual_info_classif, k=topN_features)
    selector.fit_transform(X_train_ftes, y_train_ftes)
    slct_cols_idx = selector.get_support(indices=True)

    # build dataframe with all feature scores
    fte_imps = {}
    for feature, score in zip(selector.feature_names_in_, selector.scores_):
        fte_imps[feature] = score
    df_fte_imps = pd.DataFrame({"features": fte_imps.keys(), "score": fte_imps.values()})

    slct_cols = X_train_ftes.iloc[:, slct_cols_idx].columns.tolist()
    df_ftes = df_ftes[INDEX_COL + slct_cols]

    return df_ftes, df_fte_imps


def apply_vif_fte_slct(df_ftes: pd.DataFrame,
                            spine_labeled: pd.DataFrame,
                            train_test_cutoff_date: str,
                            topN_features: int,
                            vif_threshold: Union[int, float],
                            apply_mic_after_vif: bool = True) -> Tuple[pd.DataFrame,
                                                                        pd.DataFrame,
                                                                        pd.DataFrame]:
    """Select feature using the Variance Inflation Factor (VIF) method
    Potentially apply MIC after VIF

    Args:
        df_ftes (pd.DataFrame): dataframe containing the features
        spine_labeled (pd.DataFrame): spine dataset already with class labels
        train_test_cutoff_date (str): cutoff date to split dataset into train and test
        topN_features (int): amount of features to keep
        vif_threshold (Union[int, float]): threshold value to cutoff features for VIF
        apply_mic_after_vif (bool, optional): whether or not to apply MIC feature selection after VIF. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: dataframe with remaining features, 
        dataframe with VIF values and dataframe with MIC features' importance value
    """

    df_ftes_aux = df_ftes.copy() # do not change original data, it will be reused
    df_ftes_aux = df_ftes_aux.dropna()

    # df = df[df["close_time"] < train_test_cutoff_date]
    df_ftes_aux.loc[:, "const"] = 1 #VIF requires a constant col
    df_ftes_aux = df_ftes_aux.set_index(INDEX_COL)

    _vif = pd.DataFrame()
    _vif.loc[:, "features"] = df_ftes_aux.columns.copy()
    _vif.loc[:, "vif"] = [vif(df_ftes_aux.values, i) for i in range(df_ftes_aux.shape[1])]

    _vif = _vif[_vif["features"] != "const"] #drop constant col

    # get symbols below threshold
    slct_symbols = _vif[_vif["vif"] <= vif_threshold]["features"].tolist()
    df_ftes = df_ftes[INDEX_COL + slct_symbols]

    if apply_mic_after_vif:
        df_ftes, df_fte_imps = apply_mic_fte_slct(df_ftes=df_ftes,
                                                    spine_labeled=spine_labeled,
                                                    train_test_cutoff_date=train_test_cutoff_date,
                                                    topN_features=topN_features)

    else:
        df_fte_imps = pd.DataFrame()

    return df_ftes, _vif, df_fte_imps
