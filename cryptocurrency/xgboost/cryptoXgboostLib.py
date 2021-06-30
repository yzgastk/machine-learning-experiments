import os
import time
import sqlite3
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from stldecompose import decompose

import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import HomeIndicators as ind

def createPandaFrame(conn, symbol, interval, lastAdded=None):
	loadQuery = "SELECT * FROM "+symbol+"_KLINES"+interval+" ORDER BY open_time;"
	cur = conn.execute(loadQuery)
	pdFrame = pd.DataFrame(cur.fetchall())
	cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trade_number', 'tb_base_av', 'tb_quote_av']
	pdFrame.columns = cols
	pdFrame['open_time'] = pdFrame['open_time'].apply(humanDate)
	pdFrame['close_time_decomp'] = pd.to_datetime(pdFrame['close_time'].apply(lambda x: int(x)/1000))
	pdFrame['close_time'] = pdFrame['close_time'].apply(humanDate)
	return pdFrame


def humanDate(date):
	secondsDate = int(date) / 1000
	hDate = datetime.datetime.fromtimestamp(secondsDate)
	return hDate


def xgboost_func(df):
	df_close = df[['close_time', 'close']].copy()
	df_close = df_close.set_index('close_time')
	df_close.head()

	decomp = decompose(df_close, period=365)

	df['EMA_9'] = ind.getEMA(df['close'], timeperiod=9)
	df['EMA_12'] = ind.getEMA(df['close'], timeperiod=12)
	df['EMA_26'] = ind.getEMA(df['close'], timeperiod=26)
	df['SMA_5'] = ind.getSMA(df['close'], timeperiod=5)
	df['SMA_10'] = ind.getSMA(df['close'], timeperiod=10)
	df['SMA_15'] = ind.getSMA(df['close'], timeperiod=15)
	df['SMA_30'] = ind.getSMA(df['close'], timeperiod=30)

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df.close_time, y=df.EMA_9, name='EMA 9'))
	fig.add_trace(go.Scatter(x=df.close_time, y=df.SMA_5, name='SMA 5'))
	fig.add_trace(go.Scatter(x=df.close_time, y=df.SMA_10, name='SMA 10'))
	fig.add_trace(go.Scatter(x=df.close_time, y=df.SMA_15, name='SMA 15'))
	fig.add_trace(go.Scatter(x=df.close_time, y=df.SMA_30, name='SMA 30'))
	fig.add_trace(go.Scatter(x=df.close_time, y=df.close, name='Close', opacity=0.2))
	# fig.show()

	df['RSI'] = ind.getRSI(x=df['close'],timeperiod=14)

	fig = go.Figure(go.Scatter(x=df.close_time, y=df.RSI, name='RSI'))
	# fig.show()


	df['MACD'], df['MACD_signal'], _ = ind.getMACD(x=df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
	fig = make_subplots(rows=2, cols=1)
	fig.add_trace(go.Scatter(x=df.close_time, y=df.close, name='Close'), row=1, col=1)
	fig.add_trace(go.Scatter(x=df.close_time, y=df.EMA_12, name='EMA 12'))
	fig.add_trace(go.Scatter(x=df.close_time, y=df.EMA_26, name='EMA 26'))
	fig.add_trace(go.Scatter(x=df.close_time, y=df['MACD'], name='MACD'), row=2, col=1)
	fig.add_trace(go.Scatter(x=df.close_time, y=df['MACD_signal'], name='Signal line'), row=2, col=1)
	# fig.show()

	# Training
	df['close'] = df['close'].shift(-1)
	df = df.iloc[33:]
	df = df[:-1]
	df.index = range(len(df))
	print(df.head())

	test_size  = 0.15
	valid_size = 0.15

	test_split_idx  = int(df.shape[0] * (1-test_size))
	valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

	train_df  = df.loc[:valid_split_idx].copy()
	valid_df  = df.loc[valid_split_idx+1:test_split_idx].copy()
	test_df   = df.loc[test_split_idx+1:].copy()

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=train_df.close_time, y=train_df.close, name='Training'))
	fig.add_trace(go.Scatter(x=valid_df.close_time, y=valid_df.close, name='Validation'))
	fig.add_trace(go.Scatter(x=test_df.close_time,  y=test_df.close,  name='Test'))
	# fig.show()
	print(list(train_df.columns))
	drop_cols = ['open_time', 'open', 'high', 'low', 'volume', 'close_time', 'quote_asset_volume', 'trade_number', 'tb_base_av', 'tb_quote_av', 'close_time_decomp']

	train_df = train_df.drop(drop_cols, axis=1)
	valid_df = valid_df.drop(drop_cols, axis=1)
	test_df  = test_df.drop(drop_cols, axis=1)

	y_train = train_df['close'].copy()
	X_train = train_df.drop(['close'], axis=1)

	y_valid = valid_df['close'].copy()
	X_valid = valid_df.drop(['close'], axis=1)

	y_test  = test_df['close'].copy()
	X_test  = test_df.drop(['close'], axis=1)

	X_train.info()

	parameters = {
		'n_estimators': [100, 200, 300, 400],
		'learning_rate': [0.001, 0.005, 0.01, 0.05],
		'max_depth': [8, 10, 12, 15],
		'gamma': [0.001, 0.005, 0.01, 0.02],
		'random_state': [42]
	}

	eval_set = [(X_train, y_train), (X_valid, y_valid)]
	model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
	clf = GridSearchCV(model, parameters)

	clf.fit(X_train, y_train)

	print(f'Best params: {clf.best_params_}')
	print(f'Best validation score = {clf.best_score_}')

	model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
	model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

	plot_importance(model)
	plt.show()
	plt.clf()

	y_pred = model.predict(X_test)
	print(f'y_true = {np.array(y_test)[:5]}')
	print(f'y_pred = {y_pred[:5]}')
	print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')

	predicted_prices = df.loc[test_split_idx+1:].copy()
	predicted_prices['close'] = y_pred

	fig = make_subplots(rows=2, cols=1)
	fig.add_trace(go.Scatter(x=df.close_time, y=df.close, name='Truth',marker_color='LightSkyBlue'), row=1, col=1)
	fig.add_trace(go.Scatter(x=predicted_prices.close_time, y=predicted_prices.close, name='Prediction', marker_color='MediumPurple'), row=1, col=1)
	fig.add_trace(go.Scatter(x=predicted_prices.close_time, y=y_test, name='Truth', marker_color='LightSkyBlue', showlegend=False), row=2, col=1)
	fig.add_trace(go.Scatter(x=predicted_prices.close_time, y=y_pred, name='Prediction', marker_color='MediumPurple', showlegend=False), row=2, col=1)
	fig.show()


if __name__ == '__main__':
	symbol = sys.argv[1]
	interval = sys.argv[2]
	try:
		subset = sys.argv[3]
	except:
		subset = None

	asset = symbol+interval
	conn = sqlite3.connect("./data/historicData.sqlite")
	df = createPandaFrame(conn, symbol, interval)
	df.index = range(len(df))
	ts = time.time()

	xgboost_func(df)

	print(time.time() - ts)
