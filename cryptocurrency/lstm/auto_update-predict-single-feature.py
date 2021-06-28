import sys, math, sqlite3
import datetime, time, requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from math import sqrt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# import os; os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

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


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	if test.size > 0:
		test = test.reshape(test.shape[0], test.shape[1])
		test_scaled = scaler.transform(test)
	else:
		test_scaled = []
	return scaler, train_scaled, test_scaled


def fit_lstm(train, batch_size, nb_epoch, neurons, loadedModel=None):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = loadedModel
	if not loadedModel:
		model = Sequential()
		model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
		model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df


def splitData(datas, train_ratio=0.66):
	n_train = train_ratio
	train = values[:int(n_train*len(datas))]
	test = values[int(n_train*len(datas)):]
	return train, test

def splitDataBis(datas, train_ratio=0.66):
	n_train = train_ratio
	datas2 = datas[-100:]
	train = datas2[:int(n_train*len(datas2)), :]
	test = datas2[int(n_train*len(datas2)):, :]
	return train, test

def tidyUp(df):
	drop_cols = ['open_time', 'open', 'high', 'low', 'volume', 'close_time', 'quote_asset_volume', 'trade_number', 'tb_base_av', 'tb_quote_av', 'batch_number']
	return df.drop(drop_cols, axis=1)


def humanDate(date):
	secondsDate = int(date) / 1000
	hDate = datetime.datetime.fromtimestamp(secondsDate)
	return hDate.strftime('%Y-%m-%d %H:%M:%S')


def createPandaFrame(conn, interval, lastAdded=None):
	loadQuery = "SELECT * FROM KLINES"+interval+" ORDER BY open_time;"
	cur = conn.execute(loadQuery)
	pdFrame = pd.DataFrame(cur.fetchall())
	cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trade_number', 'tb_base_av', 'tb_quote_av', 'batch_number']
	pdFrame.columns = cols
	pdFrame['open_time'] = pdFrame['open_time'].apply(humanDate)
	pdFrame['close_time'] = pdFrame['close_time'].apply(humanDate)
	if lastAdded:
		pdFrame = pdFrame[lastAdded:]
	return pdFrame


def flattenUnitaryList(valuesDf):
	return np.array([ elem for singleList in valuesDf for elem in singleList])

if __name__ == '__main__':

	symbol = sys.argv[1]
	interval = sys.argv[2]

	conn = sqlite3.connect("./data/CP-"+sys.argv[1]+".sqlite")
	start_time = time.time()
	loadedModel = load_model('./data/cp-auto_update-time-series-forecasting-lstm-'+interval+'.h5')
	predictions = []
	trueValues = []
	valDates = []
	continuePredicting = True

	while continuePredicting:
		start_time = time.time()
		print("Predicting one step ahead.")
		lastOpenTime = "SELECT MAX(open_time) FROM KLINES"+interval+";"
		cursor = conn.execute(lastOpenTime)
		lastTime = int(cursor.fetchone()[0])
		mdate = int(lastTime / 1000)
		tDate = datetime.datetime.fromtimestamp(mdate)
		hDate = tDate.strftime('%Y-%m-%d %H:%M:%S')
		print("Got last time = "+hDate)
		klines = getKlines(symbol, interval=interval, startTime=lastTime, limit=10)
		toFit = []
		for kline in klines:
			print(str(lastTime)+" < "+str(kline[0]))
			if lastTime <= kline[0]:
				toFit.append(float(kline[4]))
			insertQuery = "INSERT OR REPLACE INTO KLINES"+interval+" (open_time, open, high, low, close, volume, close_time, quote_asset_volume, trade_number, tb_base_av, tb_quote_av) VALUES ('"+str(kline[0])+"', "+kline[1]+", "+kline[2]+", "+kline[3]+", "+kline[4]+", "+kline[5]+", '"+str(kline[6])+"', "+kline[7]+", "+str(kline[8])+", "+kline[9]+", "+kline[10]+");"
			conn.execute(insertQuery)

		getPredQuery = 'SELECT close FROM KLINES'+interval+' ORDER BY open_time DESC LIMIT '+str(len(toFit)+1)+';'
		cursor = conn.execute(getPredQuery)
		toFit.insert(0, cursor.fetchall()[-1][0])
		print(toFit)
		values = np.array(toFit)
		print("Values to train : "+str(values[:-1]))
		print("Testing for : "+str(values[-1]))
		trueValues.append(values[-1])
		valDates.append(humanDate(klines[-1][0]))
		stationary = difference(values)
		supervised = timeseries_to_supervised(stationary)
		print(supervised)
		train = supervised[:-1].values
		print("-- SupTrain --")
		print(train)
		test = supervised[-1:].values
		print("-- SupTest --")
		print(test)
		scaler, train_scaled, test_scaled = scale(train, test)
		lstm_model = fit_lstm(train_scaled, 1, 50, 4, loadedModel=loadedModel)

		X = test_scaled[0, 0:-1]
		yhat = forecast_lstm(lstm_model, 1, X)
		yhat = invert_scale(scaler, X, yhat)
		yhat = inverse_difference(values, yhat, len(test_scaled)+1)
		predictions.append(yhat)
		print("Predicted : "+str(yhat))
		print("--- %s seconds ---" % (time.time() - start_time))
		conn.commit()
		continuePredicting += 1
		time.sleep(60)
		if continuePredicting > 10:
			continuePredicting = False


	fig, ax = plt.subplots()
	ax.plot(valDates, trueValues)
	ax.plot(predictions)
	fig.autofmt_xdate()
	plt.show()
