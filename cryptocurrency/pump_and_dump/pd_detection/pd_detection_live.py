#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import time
import datetime
import pandas as pd
import Klines


def compute_features(kline_dictionary, btc_price, cmc_frame):
	pump_list = pd.read_csv("./data/coin-pump.csv")
	rm_key = []
	for symbol in kline_dictionary.keys():

		kline_dictionary[symbol].pdFrame['ret1'] = np.log(kline_dictionary[symbol].pdFrame.Close) - np.log(kline_dictionary[symbol].pdFrame.Close.shift(1))
		kline_dictionary[symbol].pdFrame = pd.merge(kline_dictionary[symbol].pdFrame, btc_price, how='inner', left_index=True, right_index=True)

		xs = [1, 3, 12, 24, 36, 48, 60, 72]
		ys = [3, 12, 24, 36, 48, 60, 72]

		for x in xs:
			strx = str(x)
			ret = "ret" + strx
			volf = "volf" + strx
			volbtc = "volbtc" + strx
			kline_dictionary[symbol].pdFrame[ret] = kline_dictionary[symbol].pdFrame['ret1'].rolling(x).sum()
			kline_dictionary[symbol].pdFrame[volf] = kline_dictionary[symbol].pdFrame['Volume'].rolling(x).sum()
			kline_dictionary[symbol].pdFrame[volbtc] = (kline_dictionary[symbol].pdFrame['Volume'] * kline_dictionary[symbol].pdFrame['Close']) / kline_dictionary[symbol].pdFrame['btc_price']

		for y in ys:
			stry = str(y)
			vola = "vola" + stry
			volavol = "volavol" + stry
			rtvol = "rtvol" + stry
			kline_dictionary[symbol].pdFrame[vola] = kline_dictionary[symbol].pdFrame['ret1'].rolling(
				y).std()  # * (x ** 0.5) ??? => According to finance expert this should be included BUT according to the Rshit code on github it does not seem to be there
			kline_dictionary[symbol].pdFrame[volavol] = kline_dictionary[symbol].pdFrame['volf' + str(y)].rolling(y).std()
			kline_dictionary[symbol].pdFrame[rtvol] = kline_dictionary[symbol].pdFrame['volbtc' + str(y)].rolling(y).std()

		unpaired_token = symbol#[:-3]
		pumps = pump_list.loc[pump_list['Coin'] == unpaired_token].sort_values('Date')



		ptr = cmc_frame.loc[unpaired_token]

		try:
			kline_dictionary[symbol].pdFrame['existence_time'] = ((pd.to_datetime(kline_dictionary[symbol].pdFrame.index) - pd.to_datetime(ptr["date_added"]).replace(tzinfo=None)).days)
		except KeyError:
			rm_key.append(symbol)
			continue
		except TypeError:
			ptr = cmc_frame.loc[unpaired_token].tail(1).iloc[0]
			kline_dictionary[symbol].pdFrame['existence_time'] = ((pd.to_datetime(kline_dictionary[symbol].pdFrame.index) - pd.to_datetime(ptr["date_added"]).replace(tzinfo=None)).days)

		kline_dictionary[symbol].pdFrame['market_cap'] = ptr["total_supply"]
		kline_dictionary[symbol].pdFrame['coin_rating'] = ptr["cmc_rank"]

		try:
			kline_dictionary[symbol].pdFrame['pump_count'] = pumps.groupby('Coin').size().iloc[0]
		except IndexError:
			kline_dictionary[symbol].pdFrame['pump_count'] = 0

		kline_dictionary[symbol].pdFrame['last_open_price'] = kline_dictionary[symbol].pdFrame['Open'].shift(1)
		kline_dictionary[symbol].pdFrame['symbol'] = unpaired_token
		kline_dictionary[symbol].pdFrame['label'] = 0

	for symbol in rm_key:
		del kline_dictionary[symbol]

	return rm_key


def update_features(symbol, temp_df, btc_price, cmc_frame):
	pump_list = pd.read_csv("./data/coin-pump.csv")
	xs = [1, 3, 12, 24, 36, 48, 60, 72]
	ys = [3, 12, 24, 36, 48, 60, 72]

	temp_df['ret1'] = np.log(temp_df.Close) - np.log(temp_df.Close.shift(1))
	temp_df['btc_price'] = btc_price

	for x in xs:
		strx = str(x)
		ret = "ret" + strx
		volf = "volf" + strx
		volbtc = "volbtc" + strx
		temp_df[ret] = temp_df['ret1'].rolling(x).sum()
		temp_df[volf] = temp_df['Volume'].rolling(x).sum()
		temp_df[volbtc] = (temp_df['Volume'] * temp_df['Close']) / temp_df['btc_price']

	for y in ys:
		stry = str(y)
		vola = "vola" + stry
		volavol = "volavol" + stry
		rtvol = "rtvol" + stry
		temp_df[vola] = temp_df['ret1'].rolling(y).std()
		temp_df[volavol] = temp_df['volf' + str(y)].rolling(y).std()
		temp_df[rtvol] = temp_df['volbtc' + str(y)].rolling(y).std()

	unpaired_token = symbol[:-3]
	pumps = pump_list.loc[pump_list['Coin'] == unpaired_token].sort_values('Date')

	temp_df['existence_time'] = ((pd.to_datetime(temp_df.index) - pd.to_datetime(
			cmc_frame.loc[unpaired_token]["date_added"]).replace(tzinfo=None)).days)

	temp_df['market_cap'] = cmc_frame.loc[unpaired_token]["total_supply"]
	temp_df['coin_rating'] = cmc_frame.loc[unpaired_token]["cmc_rank"]

	temp_df['pump_count'] = pumps.groupby('Coin').size().iloc[0]

	temp_df['last_open_price'] = temp_df['Open'].shift(1)
	temp_df['symbol'] = unpaired_token
	temp_df['label'] = 0

	return


if __name__ == '__main__':
	file_all_coins = open("all_coins", 'rb')
	all_coins = pickle.load(file_all_coins)

	file_model = open("./data/pd_detection_model2.pkl", "rb")
	model = pickle.load(file_model)

	unpickle = open('./data/pd_symbols.pkl', 'rb')
	symbols = pickle.load(unpickle)
	symbols = all_coins.loc[all_coins.status == "TRADING"].symbol.unique().tolist()

	unpickle = open('./data/cmc_market_data.pkl', 'rb')
	cmc_frame = pickle.load(unpickle)
	cmc_frame = cmc_frame.set_index('symbol')
	unpickle.close()

	counter = 0
	timeframe = "1h"
	timeframe_seconds = 3600 # 1 hour
	auto_stop = 24 # 1 day
	look_back = 72 # 3 days

	while counter < auto_stop:
		counter += 1

		btc_klines = Klines.Klines("BTCUSDT", timeframe, futures=False, keep_last=True)
		btc_klines.createPandaFrame(heikinAshi=False)
		btc_price = btc_klines.pdFrame.Close
		btc_price.name = "btc_price"

		kline_dictionary = {}
		for symbol in symbols:
			kline_dictionary[symbol] = Klines.Klines(symbol+"BTC", timeframe, futures=False, keep_last=True)
			kline_dictionary[symbol].createPandaFrame(heikinAshi=False)

		rm_key = compute_features(kline_dictionary, btc_price, cmc_frame)
		for symbol in rm_key:
			symbols.remove(symbol)

		seer_buffer = {}
		for symbol in kline_dictionary:
			last_row = kline_dictionary[symbol].pdFrame.tail(2).head(1)
			cols_to_drop = ["Open", "High", "Low", "Close", "Volume", "symbol", "label"]
			feature_input = last_row.drop(cols_to_drop, axis=1)
			if feature_input.isnull().values.any():
				continue
			oracle = model.predict_proba(feature_input)

			seer_buffer[symbol] = {}
			seer_buffer[symbol]["pos"] = oracle[0][1]
			seer_buffer[symbol]["neg"] = oracle[0][0]
			pass

		unsorted_predicitions = pd.DataFrame(seer_buffer).transpose().sort_values("pos")
		negs = unsorted_predicitions.loc[unsorted_predicitions.neg >= 0.5]
		poss = unsorted_predicitions.loc[unsorted_predicitions.neg < 0.5]

		print("### NEGATIVE "+"#"*30)
		for line in negs.iterrows():
			print(line[0]+": "+str(line[1]["pos"])+"|"+str(line[1]["neg"]))

		print("### POSITIVE " + "#" * 30)
		for line in poss.iterrows():
			print(line[0]+": "+str(line[1]["pos"])+"|"+str(line[1]["neg"]))

		time_delta = timeframe_seconds - (int(datetime.datetime.now().strftime('%s')) % timeframe_seconds)
		print("Timedelta = "+str(time_delta)+" seconds")
		time.sleep(time_delta + 1)
