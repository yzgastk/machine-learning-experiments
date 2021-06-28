# IMPORTS
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, LeakyReLU

# Load information about the stock
stonks = 'fdx.us.txt'
df = pd.read_csv('/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/'+stonks)
print(df.info())

# Compute labels automatically based on the method given in the paper
# If value is greater 10 day ahead then label current day "buy (1)" otherwise "sell (0)"
df['Open10'] = df['Open'].shift(periods=-10)
df = df.dropna()
df['label'] = np.where(df['Open'] < df['Open10'], 1, 0)


# Drop useless columns, keeping only OHLCV
dropCols= ['Date', 'OpenInt', 'Open10'] # 'Open', 'High', 'Low', 'Close', 'Volume'
df = df.drop(labels=dropCols, axis=1)
print(df.head())

# Computing split index
x, _ = df.shape
splitCoeff = 0.80
splitRow = int(x * splitCoeff)
dfNum = df.to_numpy()

# Separating training data from testing data
trainX = dfNum[:splitRow, :-1]
trainY = dfNum[:splitRow, -1]
testX = dfNum[splitRow:, :-1]
testY = dfNum[splitRow:, -1]

# Showing price figure for 'open' prices
xPlot = range(0, x)
plt.figure()
plt.title('Data Separation')
plt.grid(True)
plt.ylabel('Open Price')
plt.plot(xPlot[:splitRow], trainX[:,0], 'blue', label='Train data')
plt.plot(xPlot[splitRow:], testX[:,0], 'red', label='Test data')
plt.legend()
plt.show()
plt.close()

# Input normalization
norm = MinMaxScaler()
trainX = norm.fit_transform(trainX)

# Reshaping data to get a 3D tensor for Conv1D
trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
testX = testX.reshape(testX.shape[0], testX.shape[1], 1)

# Parameters
nFeatures = trainX.shape[1]
epochs = 20
batchSize = 1000
nOutput = 1
kernelSize = 1

# Model Building
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=kernelSize, padding='same', activation='relu', input_shape=(nFeatures, 1)))
model.add(Conv1D(filters=64, kernel_size=kernelSize, padding='same'))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling1D(pool_size=(1)))
model.add(Conv1D(filters=128, kernel_size=kernelSize, padding='same'))
model.add(LeakyReLU(alpha=0.01))
model.add(Flatten())
model.add(Dense(256,))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.8))
model.add(Dense(nOutput, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Fit the model
fitReturn = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, batch_size=batchSize, verbose=1)

plt.title('Loss')
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.plot(fitReturn.history['loss'], 'blue', label='Train Loss')
plt.plot(fitReturn.history['val_loss'], 'red', label='Test Loss')
plt.legend()
plt.show()
plt.close()

plt.title('Accuracy')
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.plot(fitReturn.history['accuracy'], 'blue', label='Train Accuracy')
plt.plot(fitReturn.history['val_accuracy'], 'red', label='Test Accuracy')
plt.legend()
plt.show()
plt.close()

# Predictions
predictY = model.predict_classes(testX, verbose=0)
predictY = predictY[:, 0]

# Basic Counting
testY0 = (testY == 0).sum()
testY1 = (testY == 1).sum()
print("Test Set - Sell signal : "+str(testY0))
print("Test Set - Buy signal  : "+str(testY1))
print("="*40)
predictY0 = (predictY == 0).sum()
predictY1 = (predictY == 1).sum()
print("Predicted - Sell signal : "+str(predictY0))
print("Predicted - Buy signal  : "+str(predictY1))

# Computing Scores
accuracy = accuracy_score(testY, predictY)
precision = precision_score(testY, predictY)
recall = recall_score(testY, predictY)
f1 = f1_score(testY, predictY)
matrix = confusion_matrix(testY, predictY)
print('Accuracy: '+str(accuracy))
print('Precision: '+str(precision))
print('Recall: '+str(recall))
print('F1 Score: '+str(f1))
print(matrix)
