#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
import pickle
import time

from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def train_loop(conn_data, model, cv, split_amount, line_count):
    start = time.time()
    print("Round(" + str(line_count / split_amount) + "): [" + str(line_count) + "-" + str(line_count + split_amount) + "]")
    feature_input = pd.read_sql("SELECT * FROM aggregated_data ORDER BY close_time ASC LIMIT " + str(split_amount) + " OFFSET " + str(line_count) + ";", conn_data)

    columns = ["open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trade_number",
               "tb_base_av",
               "tb_quote_av", "label", "listing_date", "existence_time", "ret1", "amplitude_intra", "pump_count",
               "btc_price",
               "market_cap", "volf1", "volbtc1", "ret3", "volf3", "volbtc3", "ret12", "volf12", "volbtc12", "ret24",
               "volf24",
               "volbtc24", "ret36", "volf36", "volbtc36", "ret48", "volf48", "volbtc48", "ret60", "volf60", "volbtc60",
               "ret72",
               "volf72", "volbtc72", "vola3", "volavol3", "rtvol3", "vola12", "volavol12", "rtvol12", "vola24",
               "volavol24",
               "rtvol24", "vola36", "volavol36", "rtvol36", "vola48", "volavol48", "rtvol48", "vola60", "volavol60",
               "rtvol60",
               "vola72", "volavol72", "rtvol72", "last_open_price", "symbol"]

    feature_input.columns = columns
    cols_to_drop = ["open", "high", "low", "listing_date", "close", "volume", "close_time", "tb_base_av", "tb_quote_av",
                    "symbol"]
    feature_input = feature_input.drop(cols_to_drop, axis=1)

    x_train = feature_input.dropna()
    y_train = x_train.label
    x_train.drop("label", inplace=True, axis=1)

    print("Starting fitting for round " + str(line_count / split_amount) + ".")
    for train_index, test_index in cv.split(x_train, y_train):
        model.fit(x_train.iloc[train_index], y_train.iloc[train_index])
        print(str((time.time() - start) / 60) + " minutes")

    line_count += split_amount


if __name__ == '__main__':
    conn_data = sqlite3.connect("./data/spike_data.sqlite")
    split_amount = 300000
    base_index = 0
    line_count = 0
    db_size_req = pd.read_sql("SELECT COUNT(*) FROM aggregated_data;", conn_data)
    db_size = db_size_req.iloc[0][0]
    stop = int(db_size/split_amount)

    model = RandomForestClassifier(n_jobs=-1, verbose=1)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


    while int(line_count/split_amount) <= 1:
        train_loop(conn_data, model, split_amount, line_count)

    file_model = open("./data/model_spike_2.pkl", "wb")
    pickle.dump(model, file_model)
    file_model.close()
