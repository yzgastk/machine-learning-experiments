#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import pickle
import sqlite3
import datetime

import pandas as pd
import numpy as np


def createPandaFrame(conn, tablename, cols, convert_date=True):
    loadQuery = "SELECT * FROM " + tablename + " ORDER BY open_time;"
    cur = conn.execute(loadQuery)
    pdFrame = pd.DataFrame(cur.fetchall())
    pdFrame.columns = cols

    if convert_date:
        pdFrame['open_time'] = pdFrame['open_time'].apply(humanDate)
        pdFrame['close_time_decomp'] = pd.to_datetime(pdFrame['close_time'].apply(lambda x: int(x) / 1000))
        pdFrame['close_time'] = pdFrame['close_time'].apply(humanDate)
    else:
        pdFrame['open_time'] = pd.to_datetime(pdFrame['open_time'])
        pdFrame['close_time'] = pd.to_datetime(pdFrame['close_time'])


    return pdFrame


def humanDate(date):
    secondsDate = int(date) / 1000
    hDate = datetime.datetime.fromtimestamp(secondsDate)
    return hDate


def compute_features(conn_features, symbols, interval):
    for symbol in symbols:
        print("== Computing features for "+symbol)
        conn = sqlite3.connect("./data/pd_crypto.sqlite")
        cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                'trade_number', 'tb_base_av', 'tb_quote_av', 'label']
        temp_df = createPandaFrame(conn, symbol + "_KLINES" + interval, cols)
        temp_df['ret1'] = np.log(temp_df.close) - np.log(temp_df.close.shift(1))
        temp_df = pd.merge(temp_df, btcPrice, how='inner', left_index=True, right_index=True)

        xs = [1, 3, 12, 24, 36, 48, 60, 72]
        ys = [3, 12, 24, 36, 48, 60, 72]

        for x in xs:
            strx = str(x)
            ret = "ret" + strx
            volf = "volf" + strx
            volbtc = "volbtc" + strx
            temp_df[ret] = temp_df['ret1'].rolling(x).sum()
            temp_df[volf] = temp_df['volume'].rolling(x).sum()
            temp_df[volbtc] = (temp_df['volume'] * temp_df['close']) / temp_df['btc_price']

        for y in ys:
            stry = str(y)
            vola = "vola" + stry
            volavol = "volavol" + stry
            rtvol = "rtvol" + stry
            temp_df[vola] = temp_df['ret1'].rolling(y).std()
            temp_df[volavol] = temp_df['volf'+str(y)].rolling(y).std()
            temp_df[rtvol] = temp_df['volbtc'+str(y)].rolling(y).std()

        temp_df.to_sql(symbol, conn_features, if_exists='replace', index=False)
        conn_features.commit()
    conn_features.close()

def create_pump_database():
    interval = "1h"

    unpickle = open('./data/pd_symbols.pkl', 'rb')
    symbols = pickle.load(unpickle)

    conn = sqlite3.connect("./data/pd_detection.sqlite")
    cols = ['open_time', 'open', 'high', 'low', 'btc_price', 'volume', 'close_time', 'quote_asset_volume', 'trade_number', 'tb_base_av', 'tb_quote_av', 'label']
    btcPrice = createPandaFrame(conn, "BTCUSDT" + "_KLINES" + interval, cols)
    btcPrice = btcPrice['btc_price']
    conn.close()

    pump_list = pd.read_csv("./data/binance_pump_only.txt")
    pump_list2 = pd.read_csv("./data/coin-pump.csv")

    unpickle = open('./data/cmc_market_data.pkl', 'rb')
    cmc_frame = pickle.load(unpickle)
    cmc_frame = cmc_frame.set_index('symbol')

    conn_features = sqlite3.connect("./data/pd_features.sqlite")
    conn_pump_input = sqlite3.connect("./data/pump_input.sqlite")
    pump_list2['date'] = pump_list2['Date'] + " " + pump_list2['Time']

    mega_dict = {}
    basic_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                  'trade_number', 'tb_base_av', 'tb_quote_av', 'label', 'close_time_decomp', ]
    cols = ['ret1', 'btc_price', 'volf1',
            'volbtc1', 'ret3', 'volf3', 'volbtc3', 'ret12', 'volf12', 'volbtc12', 'ret24', 'volf24', 'volbtc24',
            'ret36', 'volf36', 'volbtc36', 'ret48', 'volf48', 'volbtc48', 'ret60', 'volf60', 'volbtc60', 'ret72',
            'volf72', 'volbtc72', 'vola3', 'volavol3', 'rtvol3', 'vola12', 'volavol12', 'rtvol12', 'vola24',
            'volavol24', 'rtvol24', 'vola36', 'volavol36', 'rtvol36', 'vola48', 'volavol48', 'rtvol48', 'vola60',
            'volavol60',
            'rtvol60', 'vola72', 'volavol72', 'rtvol72']
    additional_cols = ['open_time', 'market_cap', 'last_open_price', 'existence_time', 'pump_count', 'coin_rating']
    non_used_col = ['withdrawal_fee', 'minimum_withdrawal', 'maximum_withdrawal', 'minimum_base_trade']

    for col in cols + additional_cols:
        mega_dict[col] = []

    symbol_count = 0
    for symbol in symbols:
        symbol_count += 1
        unpaired_token = symbol[:-3]
        temp_df = createPandaFrame(conn_features, symbol, basic_cols + cols, convert_date=False)

        temp_df.set_index('open_time', inplace=True)

        pumps = pump_list2.loc[pump_list2['Coin'] == unpaired_token].sort_values('date')
        print("Computing for : " + symbol + " (" + str(symbol_count) + ")")
        for index, pump in pumps.iterrows():
            pump_count = 0
            price_before = 0
            hour_before_pump = temp_df.loc[temp_df.index < pump['date']].tail(1)
            if hour_before_pump.empty:
                print("Pump date not in the data " + symbol + " : " + pump['date'])
                continue

            try:
                mega_dict['existence_time'].append((datetime.datetime.now(datetime.timezone.utc).replace(
                    tzinfo=None) - pd.to_datetime(cmc_frame.loc[unpaired_token]["date_added"]).replace(
                    tzinfo=None)).days)
                mega_dict['market_cap'].append(cmc_frame.loc[unpaired_token]["total_supply"])
                mega_dict['coin_rating'].append(cmc_frame.loc[unpaired_token]["cmc_rank"])
            except KeyError:
                print(symbol + " not found. Skipping...")
                continue
            except TypeError:
                print(symbol + " issue with the date. Skipping...")
                print(pd.to_datetime(cmc_frame.loc[unpaired_token]["date_added"]))  # .replace(tzinfo=None))
                continue

            for column in cols:
                mega_dict[column].append(hour_before_pump[column].iloc[0])

            pump_count += 1

            mega_dict['pump_count'].append(pump_count)
            mega_dict['last_open_price'].append(hour_before_pump['btc_price'].iloc[0])
            mega_dict['open_time'].append(pump['date'])

    pump_input = pd.DataFrame(mega_dict)
    pump_input.to_sql("feature_input", conn_pump_input, if_exists='replace', index=False)


if __name__ == '__main__':
    interval = "1h"

    unpickle = open('./data/pd_symbols.pkl', 'rb')
    symbols = pickle.load(unpickle)

    conn = sqlite3.connect("./data/pd_detection.sqlite")
    cols = ['open_time', 'open', 'high', 'low', 'btc_price', 'volume', 'close_time', 'quote_asset_volume',
            'trade_number',
            'tb_base_av', 'tb_quote_av', 'label']
    btcPrice = createPandaFrame(conn, "BTCUSDT" + "_KLINES" + interval, cols)
    btcPrice = btcPrice['btc_price']
    conn.close()

    pump_list = pd.read_csv("./data/binance_pump_only.txt")
    pump_list2 = pd.read_csv("./data/coin-pump.csv")

    unpickle = open('./data/cmc_market_data.pkl', 'rb')
    cmc_frame = pickle.load(unpickle)
    cmc_frame = cmc_frame.set_index('symbol')
    conn_features = sqlite3.connect("./data/pd_features.sqlite")
    conn_rf_input = sqlite3.connect("./data/aggregated_data_simplified.sqlite")

	pump_list['date'] = pump_list['date'] + " " + pump_list['hour']
    mega_dict = {}
    basic_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
            'trade_number', 'tb_base_av', 'tb_quote_av', 'label', 'close_time_decomp',]
    cols = [ 'ret1', 'btc_price', 'volf1',
            'volbtc1', 'ret3', 'volf3', 'volbtc3', 'ret12', 'volf12', 'volbtc12', 'ret24', 'volf24', 'volbtc24',
            'ret36', 'volf36', 'volbtc36', 'ret48', 'volf48', 'volbtc48', 'ret60', 'volf60', 'volbtc60', 'ret72',
            'volf72', 'volbtc72', 'vola3', 'volavol3', 'rtvol3', 'vola12', 'volavol12', 'rtvol12', 'vola24',
            'volavol24', 'rtvol24', 'vola36', 'volavol36', 'rtvol36', 'vola48', 'volavol48', 'rtvol48', 'vola60', 'volavol60',
            'rtvol60', 'vola72', 'volavol72', 'rtvol72']

	additional_cols = ['open_time', 'market_cap', 'last_open_price', 'existence_time', 'pump_count', 'coin_rating']
    non_used_col = ['withdrawal_fee', 'minimum_withdrawal', 'maximum_withdrawal', 'minimum_base_trade']

    for col in cols + additional_cols:
        mega_dict[col] = []

    symbol_count = 0
    for symbol in symbols:
        symbol_count += 1
        unpaired_token = symbol[:-3]
        try:
            temp_df = createPandaFrame(conn_features, symbol, basic_cols + cols, convert_date=False)
        except sqlite3.OperationalError:
            print("Fail for: "+symbol)
            continue

        temp_df.set_index('open_time', inplace=True)
        pumps = pump_list.loc[pump_list['symbol'] == unpaired_token].sort_values('date')
        print("Computing for : "+symbol+" ("+str(symbol_count)+")")

        try:
            temp_df['existence_time'] = ((pd.to_datetime(temp_df.index) - pd.to_datetime(cmc_frame.loc[unpaired_token]["date_added"]).replace(tzinfo=None)).days)
        except KeyError:
            continue
        except TypeError:
            continue
        temp_df['market_cap'] = cmc_frame.loc[unpaired_token]["total_supply"]
        temp_df['coin_rating'] = cmc_frame.loc[unpaired_token]["cmc_rank"]
        try:
            temp_df['pump_count'] = pumps.groupby('symbol').size().iloc[0]
        except IndexError:
            continue
        temp_df['last_open_price'] = temp_df['open'].shift(1)
        temp_df['symbol'] = unpaired_token

        temp_df['label'] = 0

        for index, pump in pumps.iterrows():
            pump_index = temp_df.loc[temp_df.index < pump['date']].tail(1).index
            if pump_index.empty:
                print("Pump date not in the data "+symbol+" : "+pump['date'])
                continue
            else:
                temp_df.loc[pump_index, 'label'] = 1

        temp_df.to_sql("aggregated_data", conn_rf_input, if_exists='append', index=False)
