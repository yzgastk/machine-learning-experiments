#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
import pickle

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier



if __name__ == '__main__':
    conn_data = sqlite3.connect("./data/pd_detection.sqlite")
    conn_pump_input = sqlite3.connect("./data/reduced_dataset.sqlite")
    feature_input = pd.read_sql("SELECT * FROM data_window ORDER BY open_time ASC;", conn_pump_input)

    columns = ["open","high","low","close","volume","close_time","quote_asset_volume","trade_number","tb_base_av",
               "tb_quote_av","label","close_time_decomp","ret1","btc_price","volf1","volbtc1","ret3","volf3","volbtc3",
               "ret12","volf12","volbtc12","ret24","volf24","volbtc24","ret36","volf36","volbtc36","ret48","volf48",
               "volbtc48","ret60","volf60","volbtc60","ret72","volf72","volbtc72","vola3","volavol3","rtvol3","vola12",
               "volavol12","rtvol12","vola24","volavol24","rtvol24","vola36","volavol36","rtvol36","vola48","volavol48",
               "rtvol48","vola60","volavol60","rtvol60","vola72","volavol72","rtvol72","existence_time","market_cap",
               "coin_rating","pump_count","last_open_price","symbol"]
    feature_input.columns = columns
    feature_input['close_time'] = pd.to_datetime(feature_input['close_time'])

    true_label = feature_input[feature_input["label"] == 1]
    false_label = feature_input[feature_input["label"] == 0]
    print(true_label.shape)
    print(false_label.shape)

    total_pump = feature_input.shape[0]
    count_split = int(total_pump / 3)

    cols_to_drop = ["open","high","low","close","volume","close_time","quote_asset_volume","trade_number","tb_base_av", "tb_quote_av","close_time_decomp", "symbol"]
    feature_input = feature_input.drop(cols_to_drop, axis=1)

    x_train = feature_input.dropna()
    y_train = x_train.label
    x_train.drop("label", inplace=True, axis=1)

    x_validate = feature_input[count_split:(count_split*2)].dropna()
    y_validate = x_validate.label
    x_validate.drop("label", inplace=True, axis=1)

    x_test = feature_input[(count_split*2):].dropna()
    y_test = x_test.label
    x_test.drop("label", inplace=True, axis=1)


    model = RandomForestClassifier(n_jobs=-1, verbose=1)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    for train_index, test_index in cv.split(x_train, y_train):
        model.fit(x_train.iloc[train_index], y_train.iloc[train_index])

    file_model = open("./data/model.pkl", "wb")
    pickle.dump(file_model, model)
    file_model.close()
