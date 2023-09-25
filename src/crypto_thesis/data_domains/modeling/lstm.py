# -*- coding: utf-8 -*-

import logging
import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.engine.sequential import Sequential
from keras.layers import LSTM, BatchNormalization, Dense
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.metrics import accuracy_score, confusion_matrix

TARGET_COL = ["label"]
# these cols were useful so far, but not anymore
INDEX_COL = "window_nbr"

logger = logging.getLogger(__name__)

# attempt to get reproducible results
RANDOM_STATE = 0
SHUFFLE = False
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def lstm_model_fit(master_table_train: pd.DataFrame,
                master_table_test: pd.DataFrame,
                seq_length: int) -> Tuple[Sequential, pd.DataFrame]:
    """Fits the LSTM model classifier

    Args:
        master_table (pd.DataFrame): dataframe representing the master table
        train_test_cutoff_date (str): cutoff date for train/test split
        seq_length (int): length of the sequence to be predicted

    Returns:
        Tuple[Sequential, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: trained
        model object, train epoch's history, features train, target train, features test and target test, respectively
    """

    master_table_train = master_table_train.set_index(INDEX_COL)
    master_table_test = master_table_test.set_index(INDEX_COL)

    X_train, y_train = master_table_train.drop(columns=TARGET_COL), master_table_train[TARGET_COL]
    X_test, y_test = master_table_test.drop(columns=TARGET_COL), master_table_test[TARGET_COL]

    X_train_scaled_seq, y_train_scaled_seq = _build_lstm_timestamps_seq(X=X_train,
                                                                        y=y_train,
                                                                        seq_length=seq_length)
    X_test_scaled_seq, y_test_scaled_seq = _build_lstm_timestamps_seq(X=X_test,
                                                                        y=y_test,
                                                                        seq_length=seq_length)

    M_TRAIN = X_train_scaled_seq.shape[0]           # number of training examples (2D)
    M_TEST = X_test_scaled_seq.shape[0]             # number of test examples (2D),full=X_test.shape[0]
    BATCH = 16                          # batch size
    EPOCH = 50                   # number of epochs
    model = _create_lstm_model(X_train_scaled_seq=X_train_scaled_seq, seq_length=seq_length)

    # Define a learning rate decay method:
    lr_decay = ReduceLROnPlateau(
                monitor='loss',
                patience=1,
                verbose=0,
                factor=0.5,
                min_lr=1e-8)

    # Define Early Stopping:
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0,
                            patience=EPOCH, verbose=1, mode='auto',
                            baseline=0, restore_best_weights=True)

    train_history = model.fit(X_train_scaled_seq, y_train_scaled_seq,
                        epochs=EPOCH,
                        batch_size=BATCH,
                        validation_split=0.0,
                        validation_data=(X_test_scaled_seq[:M_TEST], y_test_scaled_seq[:M_TEST]),
                        shuffle=SHUFFLE, verbose=0,
                        callbacks=[lr_decay, early_stop])

    lstm_epoch_train_history = pd.DataFrame.from_dict(train_history.history)
    lstm_epoch_train_history.index = list(range(1, EPOCH+1))
    lstm_epoch_train_history.index.name = "epoch"

    # saving X_train_scaled instead of X_train_scaled_seq because the first one is easy to interpret
    # changing from the first to the second one is as simple as applying the
    # function `_build_lstm_timestamps_seq`
    return model, lstm_epoch_train_history


def _create_lstm_model(X_train_scaled_seq: pd.DataFrame, seq_length: int) -> Sequential:

    # parameters
    LAYERS = [20, 20, 20, 1] #[10, 10, 10, 1]                # number of units in hidden and output layers
    N = X_train_scaled_seq.shape[2]                 # number of features
    LR = 0.0005 #0.0005                            # learning rate of the gradient descent
    LAMBD = 0.005 #0.001                         # lambda in L2 regularizaion
    DP = 0.0 #0.0                             # dropout rate
    RDP = 0.0 #0.0                            # recurrent dropout rate

    # model
    model = Sequential()
    model.add(LSTM(
        input_shape=(seq_length, N),
        units=LAYERS[0],
        activation='tanh',
        recurrent_activation='hard_sigmoid',
        kernel_regularizer=l2(LAMBD),
        recurrent_regularizer=l2(LAMBD),
        dropout=DP,
        recurrent_dropout=RDP,
        return_sequences=True,
        return_state=False,
        stateful=False,
        unroll=False
                ))
    model.add(BatchNormalization())
    model.add(LSTM(
        units=LAYERS[1],
        activation='tanh',
        recurrent_activation='hard_sigmoid',
        kernel_regularizer=l2(LAMBD),
        recurrent_regularizer=l2(LAMBD),
        dropout=DP,
        recurrent_dropout=RDP,
        return_sequences=True,
        return_state=False,
        stateful=False,
        unroll=False
                ))
    model.add(BatchNormalization())
    model.add(LSTM(
        units=LAYERS[2],
        activation='tanh',
        recurrent_activation='hard_sigmoid',
        kernel_regularizer=l2(LAMBD),
        recurrent_regularizer=l2(LAMBD),
        dropout=DP,
        recurrent_dropout=RDP,
        return_sequences=False,
        return_state=False,
        stateful=False,
        unroll=False
                ))
    model.add(BatchNormalization())
    model.add(Dense(
        units=LAYERS[3],
        activation='sigmoid'))

    # Compile the model with Adam optimizer
    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy'],
        optimizer="Adam")

    return model


def lstm_model_predict(model: Sequential,
                        master_table_test: pd.DataFrame,
                        seq_length: int) -> pd.DataFrame:
    """LSTM model prediction

    Args:
        model (Sequential): trained model object
        X_test_scaled (pd.DataFrame): dataframe with features test
        y_test (pd.DataFrame): dataframe with target test
        seq_length (int): length of the sequence to be predicted

    Returns:
        pd.DataFrame: dataframe with model's prediction
    """

    master_table_test = master_table_test.set_index(INDEX_COL)
    X_test, y_test = master_table_test.drop(columns=TARGET_COL), master_table_test[TARGET_COL]

    idxs = X_test.index.tolist()
    X_test_scaled_seq, _ = _build_lstm_timestamps_seq(X=X_test, y=y_test, seq_length=seq_length)
    M_TEST = X_test_scaled_seq.shape[0]

    predict_probas = model.predict(x=X_test_scaled_seq, batch_size=M_TEST, verbose=1)

    y_pred = []
    for predict_proba in predict_probas:
        if predict_proba > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)

    return pd.DataFrame(data={"y_pred": y_pred}, index=idxs)


def lstm_model_reporting(model: Sequential,
                        master_table_test: pd.DataFrame,
                        y_pred: pd.DataFrame,
                        model_data_interval: str,
                        spine_preproc_params: Dict[str, Any],
                        spine_label_params: Dict[str, Any],
                        train_test_cutoff_date: str,
                        slct_topN_features: int,
                        min_years_existence: int) -> pd.DataFrame:
    """LSTM model reporting

    Args:
        model (Sequential): LSTM trained classifier
        X_test (pd.DataFrame): dataframe with features test
        y_test (pd.DataFrame): dataframe with target test
        y_pred (pd.DataFrame): dataframe with model's prediction
        master_table (pd.DataFrame): dataframe representing the master table
        model_data_interval (str): interval which the raw data was collected
        spine_preproc_params (Dict[str, Any]): parameters for spine pre-processing
        spine_label_params (Dict[str, Any]): parameters for labeling the target
        train_test_cutoff_date (str): date to cutoff datasets into train and test
        slct_topN_features (int): amount of features to keep
        min_years_existence (int): minimum time, in years, for ticker existence

    Returns:
        pd.DataFrame: dataframe with model's metrics
    """

    master_table_test = master_table_test.set_index(INDEX_COL)
    X_test, y_test = master_table_test.drop(columns=TARGET_COL), master_table_test[TARGET_COL]

    # get model's accuracy
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)

    # get model's parameters
    # params = model.get_xgb_params()

    # get model's probability
    # idxs = X_test.index.tolist()
    # probas = model.predict_proba(X_test)
    # probas_df = pd.DataFrame(data=probas, index=idxs, columns=["proba_label_0", "proba_label_1"])

    # get features' importance
    # fte_imps = model.get_booster().get_score(importance_type="weight")

    # get label class balance
    label_class_balance = (y_test["label"].value_counts() / y_test.shape[0]).to_dict()

    # get selected tickers
    tickers = []
    for col in X_test.columns.tolist():
        splitted = col.split("__")
        try:
            tickers.append(splitted[1])
        except IndexError:
            pass
    tickers = list(set(tickers))

    # confusion matrix
    # normalize = all means over rows and columns
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred, normalize="all")

    reporting_df = pd.DataFrame({"model_accuracy": acc,
                                # "model_params": str(params),
                                "data_interval_collect": model_data_interval,
                                # "test_probas": str(probas_df.to_dict(orient="index")),
                                # "fte_importance": str(fte_imps),
                                "target_name": spine_preproc_params["target_name"],
                                "volume_bar_size": spine_preproc_params["volume_bar_size"],
                                "bar_ahead_predict": spine_preproc_params["bar_ahead_predict"],
                                "labeling_tau": spine_label_params["tau"],
                                "train_test_cutoff_date": train_test_cutoff_date,
                                # "model_params": str(model_params),
                                "label_class_balance": str(label_class_balance),
                                "topN_features_slct_qty": slct_topN_features,
                                "selected_tickers": str({"tickers": tickers}),
                                "min_historical_data_window_years": min_years_existence,
                                "confusion_matrix": repr(cm)
                                }, index=[0])

    return reporting_df


def _build_lstm_timestamps_seq(X: pd.DataFrame, y: pd.DataFrame, seq_length: int) -> Tuple[np.array, np.array]:
    """Build features and target prediction array for a given size of sequence

    Args:
        X (pd.DataFrame): dataframe with features
        y (pd.DataFrame): dataframe with target
        seq_length (int): length of the sequence to be predicted

    Returns:
        Tuple[np.array, np.array]: arrays of features and target, respectively, considering the prediction horizon
    """

    _X, _y = [], []

    for i in range(y.shape[0] - (seq_length-1)):
        _X.append(X.iloc[i:i+seq_length].values)
        _y.append(y.iloc[i + (seq_length-1)])

    _X, _y = np.array(_X), np.array(_y).reshape(-1,1)

    return _X, _y
