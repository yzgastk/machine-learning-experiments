#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import sys
import pickle as rick
import pandas as pd
import sqlite3 as s3

from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    conn_data = s3.connect("./data/spike_data.sqlite")
    file_model = open("./data/model_spike_temp.pkl", "rb")
    model = rick.load(file_model)
    symbol = sys.argv[1]
    split_amount = 100000

    feature_input = pd.read_sql("SELECT * FROM aggregated_data WHERE symbol = '"+symbol+"' ORDER BY close_time ASC LIMIT "+str(split_amount)+";", conn_data)
    columns = ["open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trade_number", "tb_base_av",
               "tb_quote_av", "label", "listing_date", "existence_time", "ret1", "amplitude_intra", "pump_count", "btc_price",
               "market_cap", "volf1", "volbtc1", "ret3", "volf3", "volbtc3", "ret12", "volf12", "volbtc12", "ret24", "volf24",
               "volbtc24", "ret36", "volf36", "volbtc36", "ret48", "volf48", "volbtc48", "ret60", "volf60", "volbtc60", "ret72",
               "volf72", "volbtc72", "vola3", "volavol3", "rtvol3", "vola12", "volavol12", "rtvol12", "vola24", "volavol24",
               "rtvol24", "vola36", "volavol36", "rtvol36", "vola48", "volavol48", "rtvol48", "vola60", "volavol60", "rtvol60",
               "vola72", "volavol72", "rtvol72", "last_open_price", "symbol"]

    feature_input.columns = columns
    cols_to_drop = ["open","high","low", "listing_date", "close","volume","close_time", "tb_base_av", "tb_quote_av", "symbol"]
    feature_input = feature_input.drop(cols_to_drop, axis=1)

    print(feature_input[feature_input.label == 1].shape)

    x_test = feature_input.dropna()
    y_test = x_test.label
    x_test.drop("label", inplace=True, axis=1)

    predictions = model.predict(x_test)
    conf_mat = confusion_matrix(y_test.to_list(), predictions)
    print(conf_mat)
