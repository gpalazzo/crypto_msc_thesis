# -*- coding: utf-8 -*-

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.engine.sequential import Sequential
from keras.layers import LSTM, BatchNormalization, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

TARGET_COL = ["label"]
# these cols were useful so far, but not anymore
INDEX_COL = "window_nbr"


def lstm_model_fit(master_table: pd.DataFrame,
                    train_test_cutoff_date: str,
                    model_params: Dict[str, Any],
                    seq_length: int) -> Tuple[Sequential, pd.DataFrame,
                                                pd.DataFrame, pd.DataFrame,
                                                pd.DataFrame, pd.DataFrame]:

    # model adjustment: labeling with 0 and 1
    master_table = master_table.replace({"top": 1, "bottom": 0})
    master_table = master_table.set_index(INDEX_COL)

    # split into train and test considering time series logic
    master_table_train = master_table[master_table["close_time"] < train_test_cutoff_date]
    master_table_test = master_table[master_table["close_time"] >= train_test_cutoff_date]

    # drop not useful columns
    master_table_train = master_table_train.drop(columns=["close_time"])
    master_table_test = master_table_test.drop(columns=["close_time"])

    # split into train and test
    X_train = master_table_train.drop(columns=TARGET_COL)
    y_train = master_table_train[TARGET_COL]
    X_test = master_table_test.drop(columns=TARGET_COL)
    y_test = master_table_test[TARGET_COL]

    # scaling features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(data=scaler.fit_transform(X_train.values),
                                    index=X_train.index,
                                    columns=X_train.columns
                                )
    X_test_scaled = pd.DataFrame(data=scaler.fit_transform(X_test.values),
                                    index=X_test.index,
                                    columns=X_test.columns
                                )

    X_train_scaled_seq, y_train_scaled_seq = _build_lstm_timestamps_seq(y=y_train,
                                                                        X=X_train_scaled,
                                                                        seq_length=seq_length)
    X_test_scaled_seq, y_test_scaled_seq = _build_lstm_timestamps_seq(y=y_test,
                                                                        X=X_test_scaled,
                                                                        seq_length=seq_length)

    # parameters
    LAYERS = [8, 8, 8, 1]                # number of units in hidden and output layers
    M_TRAIN = X_train_scaled_seq.shape[0]           # number of training examples (2D)
    M_TEST = X_test_scaled_seq.shape[0]             # number of test examples (2D),full=X_test.shape[0]
    N = X_train_scaled_seq.shape[2]                 # number of features
    BATCH = M_TRAIN                          # batch size
    EPOCH = 50                           # number of epochs
    LR = 5e-2                            # learning rate of the gradient descent
    LAMBD = 3e-2                         # lambda in L2 regularizaion
    DP = 0.0                             # dropout rate
    RDP = 0.0                            # recurrent dropout rate

    # model
    model = Sequential()
    model.add(LSTM(input_shape=(seq_length, N), units=LAYERS[0],
                activation='tanh', recurrent_activation='hard_sigmoid',
                kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                dropout=DP, recurrent_dropout=RDP,
                return_sequences=True, return_state=False,
                stateful=False, unroll=False
                ))
    model.add(BatchNormalization())
    model.add(LSTM(units=LAYERS[1],
                activation='tanh', recurrent_activation='hard_sigmoid',
                kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                dropout=DP, recurrent_dropout=RDP,
                return_sequences=True, return_state=False,
                stateful=False, unroll=False
                ))
    model.add(BatchNormalization())
    model.add(LSTM(units=LAYERS[2],
                activation='tanh', recurrent_activation='hard_sigmoid',
                kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                dropout=DP, recurrent_dropout=RDP,
                return_sequences=False, return_state=False,
                stateful=False, unroll=False
                ))
    model.add(BatchNormalization())
    model.add(Dense(units=LAYERS[3], activation='sigmoid'))

    # Compile the model with Adam optimizer
    model.compile(loss='binary_crossentropy',
                metrics=['accuracy'],
                optimizer=Adam(lr=LR))

    # Define a learning rate decay method:
    lr_decay = ReduceLROnPlateau(monitor='loss',
                                patience=1, verbose=0,
                                factor=0.5, min_lr=1e-8)

    # Define Early Stopping:
    early_stop = EarlyStopping(monitor='val_acc', min_delta=0,
                            patience=30, verbose=1, mode='auto',
                            baseline=0, restore_best_weights=True)

    train_history = model.fit(X_train_scaled_seq, y_train_scaled_seq,
                        epochs=EPOCH,
                        batch_size=BATCH,
                        validation_split=0.0,
                        validation_data=(X_test_scaled_seq[:M_TEST], y_test_scaled_seq[:M_TEST]),
                        shuffle=True,verbose=0,
                        callbacks=[lr_decay, early_stop])

    lstm_epoch_train_history = pd.DataFrame.from_dict(train_history.history)
    lstm_epoch_train_history.index = list(range(1, EPOCH+1))
    lstm_epoch_train_history.index.name = "epoch"

    # saving X_train_scaled instead of X_train_scaled_seq because the first one is easy to interpret
    # changing from the first to the second one is as simple as applying the
    # function `_build_lstm_timestamps_seq`
    return model, lstm_epoch_train_history, X_train_scaled, y_train, X_test_scaled, y_test


def lstm_model_predict(model: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:

    idxs = X_test.index.tolist()
    y_pred = model.predict(X_test)
    return pd.DataFrame(data={"y_pred": y_pred}, index=idxs)


def lstm_model_reporting(model: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_test: pd.DataFrame,
                            y_pred: pd.DataFrame,
                            master_table: pd.DataFrame,
                            model_data_interval: str,
                            spine_preproc_params: Dict[str, Any],
                            spine_label_params: Dict[str, Any],
                            train_test_cutoff_date: str,
                            model_params: Dict[str, Any],
                            slct_topN_features: int,
                            min_years_existence: int) -> pd.DataFrame:

    # get model's accuracy
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)

    # get model's parameters
    params = model.get_xgb_params()

    # get model's probability
    idxs = X_test.index.tolist()
    probas = model.predict_proba(X_test)
    probas_df = pd.DataFrame(data=probas, index=idxs, columns=["proba_label_0", "proba_label_1"])

    # get features' importance
    fte_imps = model.get_booster().get_score(importance_type="weight")

    # get label class balance
    label_class_balance = (master_table["label"].value_counts() / master_table.shape[0]).to_dict()

    # get selected tickers
    tickers = []
    for col in master_table.columns.tolist():
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
                                "model_params": str(params),
                                "data_interval_collect": model_data_interval,
                                "test_probas": str(probas_df.to_dict(orient="index")),
                                "fte_importance": str(fte_imps),
                                "target_name": spine_preproc_params["target_name"],
                                "volume_bar_size": spine_preproc_params["volume_bar_size"],
                                "bar_ahead_predict": spine_preproc_params["bar_ahead_predict"],
                                "labeling_tau": spine_label_params["tau"],
                                "train_test_cutoff_date": train_test_cutoff_date,
                                "model_params": str(model_params),
                                "label_class_balance": str(label_class_balance),
                                "topN_features_slct_qty": slct_topN_features,
                                "selected_tickers": str({"tickers": tickers}),
                                "min_historical_data_window_years": min_years_existence,
                                "confusion_matrix": repr(cm)
                                }, index=[0])

    return reporting_df


def _build_lstm_timestamps_seq(y: pd.DataFrame, X: pd.DataFrame, seq_length: int):

    _X, _y = [], []

    for i in range(y.shape[0] - (seq_length-1)):
        _X.append(X.iloc[i:i+seq_length].values)
        _y.append(y.iloc[i + (seq_length-1)])

    _X, _y = np.array(_X), np.array(_y).reshape(-1,1)

    return _X, _y
