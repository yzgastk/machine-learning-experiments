import sys, math, sqlite3, requests
import datetime, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from math import sqrt

import talib

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import os
# os.environ['KERAS_BACKEND'] = 'theano'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.layers import Dropout

def getKlines(symbol, interval="1d", startTime=None, endTime=None, limit=30):
	url = "https://api.binance.com/api/v3/klines?symbol="+symbol+"&interval="+interval

	if startTime:
		url += "&startTime="+str(startTime)
	if endTime:
		url += "&endTime="+str(endTime)

	r_json = requests.get(url+"&limit="+str(limit))
	print("The current weight is : "+str(r_json.headers['X-MBX-USED-WEIGHT']))
	if r_json.status_code == 429 or r_json.status_code == 418:
		print("Weight limit of the API excedeed with code"+str(r_json.status_code))
		sys.exit()

	if r_json.status_code != 200:
		print("Exiting because of unhandled status code from http header:")
		print("Status Code : "+str(r_json.status_code))
		sys.exit()

	return r_json.json()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def tidyUp(df):
	drop_cols = ['open_time', 'open', 'close_time', 'quote_asset_volume', 'trade_number', 'tb_base_av', 'tb_quote_av'] #, 'batch_number']
	return df.drop(drop_cols, axis=1)


def humanDate(date):
	secondsDate = int(date) / 1000
	hDate = datetime.datetime.fromtimestamp(secondsDate)
	return hDate.strftime('%Y-%m-%d %H:%M:%S')


def createPandaFrame(conn, interval, lastAdded=None):
	loadQuery = "SELECT * FROM KLINES"+interval+" ORDER BY open_time;"
	cur = conn.execute(loadQuery)
	pdFrame = pd.DataFrame(cur.fetchall())
	cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trade_number', 'tb_base_av', 'tb_quote_av'] #, 'batch_number']
	pdFrame.columns = cols
	pdFrame['open_time'] = pdFrame['open_time'].apply(humanDate)
	pdFrame['close_time'] = pdFrame['close_time'].apply(humanDate)
	return pdFrame

def addIndicators(pdFrame, indList=[]):
	if 'rsi' in indList:
		pdFrame['rsi'] = talib.RSI(pdFrame['close'], timeperiod=14)

	if 'sma' in indList:
			pdFrame['sma'] = talib.EMA(pdFrame['close'], timeperiod=7)

	if 'ema5' in indList:
		print("=====")
		print(pdFrame['close'])
		print("=====")
		pdFrame['ema5'] = talib.EMA(pdFrame['close'], timeperiod=5)

	if 'ema20' in indList:
		pdFrame['ema20'] = talib.EMA(pdFrame['close'], timeperiod=20)

	if 'mfi' in indList:
		pdFrame['mfi'] = talib.MFI(pdFrame['high'], pdFrame['low'], pdFrame['close'], pdFrame['volume'], timeperiod=14)

	if 'trix' in indList:
		pdFrame['trix'] = talib.TRIX(pdFrame['close'], timeperiod=30)
	return


def getKlines(symbol, interval="1d", startTime=None, endTime=None, limit=30):
	url = "https://api.binance.com/api/v3/klines?symbol="+symbol+"&interval="+interval

	if startTime:
		url += "&startTime="+str(startTime)
	if endTime:
		url += "&endTime="+str(endTime)

	r_json = requests.get(url+"&limit="+str(limit))
	print("The current weight is : "+str(r_json.headers['X-MBX-USED-WEIGHT']))
	if r_json.status_code == 429 or r_json.status_code == 418:
		print("Weight limit of the API excedeed with code"+str(r_json.status_code))
		sys.exit()

	if r_json.status_code != 200:
		print("Exiting because of unhandled status code from http header:")
		print("Status Code : "+str(r_json.status_code))
		sys.exit()

	return r_json.json()


def fillDBPredict(symbol, interval, conn):
	lastOpenTime = "SELECT MAX(open_time) FROM KLINES"+interval+";"
	cursor = conn.execute(lastOpenTime)
	lastTime = int(cursor.fetchone()[0])
	mdate = int(lastTime / 1000)
	tDate = datetime.datetime.fromtimestamp(mdate)
	hDate = tDate.strftime('%Y-%m-%d %H:%M:%S')
	print("Got last time = "+hDate)
	klines = getKlines(symbol, interval=interval, startTime=lastTime, limit=15)

	toFit = []
	for kline in klines:
		if lastTime <= kline[0]:
			toFit.append([float(kline[2]), float(kline[3]), float(kline[4]), float(kline[5])])
		insertQuery = "INSERT OR REPLACE INTO KLINES"+interval+" (open_time, open, high, low, close, volume, close_time, quote_asset_volume, trade_number, tb_base_av, tb_quote_av) VALUES ('"+str(kline[0])+"', "+kline[1]+", "+kline[2]+", "+kline[3]+", "+kline[4]+", "+kline[5]+", '"+str(kline[6])+"', "+kline[7]+", "+str(kline[8])+", "+kline[9]+", "+kline[10]+");"
		conn.execute(insertQuery)

	return lastTime, len(toFit)

def flattenUnitaryList(valuesDf):
	return np.array([ elem for singleList in valuesDf for elem in singleList])

if __name__ == '__main__':
	start_time = time.time()
	symbol = sys.argv[1]
	interval = sys.argv[2]
	try:
		modelLoad = sys.argv[3]
	except:
		modelLoad = "compute"

	start_time = time.time()
	conn = sqlite3.connect("../../databases/"+sys.argv[1]+".sqlite")

	modelLoaded = False
	if modelLoad == "load" or modelLoad == "predict":
		loadedModel = load_model('auto_update-time-series-forecasting-lstm-'+interval+'.h5')

		modelLoaded = True
		queryLastAdded = 'SELECT lasttrained FROM FIT_INDEX WHERE symbolinterval="KLINES'+interval+'";'
		cursor = conn.execute(queryLastAdded)
		lastAdded = cursor.fetchone()[0]
		df = createPandaFrame(conn, interval, lastAdded=None)
		model = loadedModel
	else:
		lastAdded = 0
		loadedModel = None
		df = createPandaFrame(conn, interval)
	print("LastAdded = "+str(lastAdded))
	trainNumber = 100000

	predictions = []
	realvalues = []
	lenToFit = 0
	i = 0
	predictTurn = 10
	continuePredicting = True
	while continuePredicting:
		indicators = ["mfi", "trix", "ema5"]
		addIndicators(df, indicators)
		if modelLoad == "predict":
			subsetDf = df[:]
		else:
			subsetDf = df[lastAdded:(trainNumber+lastAdded)]
			print(df[trainNumber:trainNumber+1])

		print("-- subsetDf shape --")
		print(subsetDf.shape)
		dfClean = tidyUp(subsetDf)
		reorderCols = ["close", "volume"] + indicators
		dfSwap = dfClean[reorderCols]
		values = dfSwap.values
		values = values.astype('float32')
		print(values)


		scaler = MinMaxScaler(feature_range=(-1, 1))
		scaled = scaler.fit_transform(values)
		reframed = series_to_supervised(scaled, 1, 1)
		print("-- reframed shape --")
		print(reframed.shape)
		if lenToFit > 0:
			reframed = reframed[-lenToFit:]
		reframed.drop(reframed.columns[[6,7,8,9]], axis=1, inplace=True)
		values = reframed.values
		train = values[:, :]
		test = []
		train_X, train_y = train[:, :-1], train[:, -1]
		test_X, test_y = [], []
		train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
		print("-- train_X shape --")
		print(train_X.shape)
		test_X = []

		if modelLoad == "compute":
			model = Sequential()
			model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
			model.add(Dense(1))
			# model.add(LSTM(units = 50, return_sequences = True, input_shape = (train_X.shape[1], train_X.shape[2])))
			# model.add(Dropout(0.2))
			#
			# model.add(LSTM(units = 50, return_sequences = True))
			# model.add(Dropout(0.2))
			#
			# model.add(LSTM(units = 50, return_sequences = True))
			# model.add(Dropout(0.2))
			#
			# model.add(LSTM(units = 50))
			# model.add(Dropout(0.2))
			#
			# model.add(Dense(units = 1))

		model.compile(loss='mae', optimizer='adam')
			# history = model.fit(train_X, train_y, epochs=100, batch_size=21, validation_data=(test_X, test_y), verbose=2, shuffle=False)
		if modelLoad == "load" or modelLoad == "compute" or lenToFit > 0:
			if lenToFit > 0:
				added = lenToFit
			else:
				added = dfClean.shape[0]
			history = model.fit(train_X, train_y, epochs=50, batch_size=1, verbose=2, shuffle=False)

			# plt.plot(history.history['loss'], label='train')
			# plt.plot(history.history['val_loss'], label='test')
			# plt.legend()
			# plt.show()

			lastAdded = lastAdded + added
			print('INSERT OR REPLACE INTO FIT_INDEX(symbolinterval, lasttrained) VALUES ("KLINES'+interval+'", '+str(lastAdded)+');')
			queryUpdateLastAdded = 'INSERT OR REPLACE INTO FIT_INDEX(symbolinterval, lasttrained) VALUES ("KLINES'+interval+'", '+str(lastAdded)+');'
			cursor = conn.execute(queryUpdateLastAdded)
			conn.commit()
			model.save("auto_update-time-series-forecasting-lstm-"+interval+".h5")
			print("Saving model for later use.")
		print("--- %s seconds ---" % (time.time() - start_time))
		#elif modelLoad == "predict":
		#	model.compile(loss='mae', optimizer='adam')
		#else:
		#	sys.exit()

		train_Xb = train_X[-1:]
		yhat = model.predict(train_Xb)
		train_Xb = train_Xb.reshape((train_Xb.shape[0], train_Xb.shape[2]))
		inv_yhat = np.concatenate((yhat, train_Xb[:, 1:]), axis=1)
		inv_yhat = scaler.inverse_transform(inv_yhat)
		inv_yhat = inv_yhat[:,0]

		train_y = train_y[-1:]
		train_y = train_y.reshape((len(train_y), 1))
		inv_y = np.concatenate((train_y, train_Xb[:, 1:]), axis=1)
		inv_y = scaler.inverse_transform(inv_y)
		inv_y = inv_y[-1:,0]

		predictions.append(inv_yhat[0])
		realvalues.append(inv_y[0])
		print(realvalues+[np.nan])
		print([np.nan]+predictions)

		if modelLoad == "load" or modelLoad == "compute" or i > predictTurn:
			continuePredicting = False
		else:
			lastTime, lenToFit = fillDBPredict(symbol, interval, conn)
			df = createPandaFrame(conn, interval)
			i += 1
			time.sleep(300)

	# import IPython; IPython.embed()
	if i > 0:
		# plt.plot(valDates, trueValues)
		plt.plot(realvalues+[np.nan])
		plt.plot([np.nan]+predictions)
		# plt.xticks(rotation=90)
		plt.show()
