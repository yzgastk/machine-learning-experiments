import sys, math, sqlite3
import datetime, time
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
	if test:
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
	drop_cols = ['open_time', 'open', 'high', 'low', 'volume', 'close_time', 'quote_asset_volume', 'trade_number', 'tb_base_av', 'tb_quote_av'] #, 'batch_number']
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
	if lastAdded:
		pdFrame = pdFrame[lastAdded:]
	return pdFrame


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
	if modelLoad == "load":
		loadedModel = load_model('auto_update-time-series-forecasting-lstm-'+interval+'.h5')
		loadedModel.summary()
		for layer in loadedModel.layers: print(layer.get_config(), layer.get_weights())
		modelLoaded = True
		queryLastAdded = 'SELECT lasttrained FROM FIT_INDEX WHERE symbolinterval="KLINES'+interval+'";'
		cursor = conn.execute(queryLastAdded)
		lastAdded = cursor.fetchone()[0]
		df = createPandaFrame(conn, interval, lastAdded=lastAdded)
	else:
		lastAdded = 0
		loadedModel = None
		df = createPandaFrame(conn, interval)
	print("LastAdded = "+str(lastAdded))
	trainNumber = 15000
	subsetDf = df[:trainNumber]
	dfClean = tidyUp(subsetDf)
	valuesDf = dfClean.values
	values = flattenUnitaryList(valuesDf)

	predictions = list()
	stationary = difference(values)
	supervised = timeseries_to_supervised(stationary)

	train = supervised.values
	test = []
	scaler, train_scaled, test_scaled = scale(train, test)

	lstm_model = fit_lstm(train_scaled, 1, 200, 100, loadedModel=loadedModel)

	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	train_predicted = lstm_model.predict(train_reshaped, batch_size=1)
	lastAdded = lastAdded + trainNumber
	print('INSERT OR REPLACE INTO FIT_INDEX(symbolinterval, lasttrained) VALUES ("KLINES'+interval+'", '+str(lastAdded)+');')
	queryUpdateLastAdded = 'INSERT OR REPLACE INTO FIT_INDEX(symbolinterval, lasttrained) VALUES ("KLINES'+interval+'", '+str(lastAdded)+');'
	cursor = conn.execute(queryUpdateLastAdded)
	conn.commit()
	lstm_model.save("auto_update-time-series-forecasting-lstm-"+interval+".h5")
	print("Saving model for later use.")
	print("--- %s seconds ---" % (time.time() - start_time))
