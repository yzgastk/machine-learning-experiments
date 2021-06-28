import pandas as pd
from numpy.random import seed

from sklearn.model_selection import KFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

baseSeed = 777
from tensorflow import random
from numpy.random import seed
seed(baseSeed)
random.set_seed(baseSeed)

trainFeatures        = pd.read_csv('./databases/lish-moa/train_features.csv')
trainTargetScored    = pd.read_csv('./databases/lish-moa/train_targets_scored.csv')
testFeatures         = pd.read_csv('./databases/lish-moa/test_features.csv')
sampleSubmission     = pd.read_csv('./databases/lish-moa/sample_submission.csv')

trainFeatures['cp_type'] = trainFeatures['cp_type'].map({'trt_cp':0, 'ctl_vehicle':1})
trainFeatures['cp_dose'] = trainFeatures['cp_dose'].map({'D1':0, 'D2':1})
trainFeatures = trainFeatures.drop(columns="sig_id")
trainTargetScored = trainTargetScored.drop(columns="sig_id")

testFeatures['cp_type'] = testFeatures['cp_type'].map({'trt_cp':0, 'ctl_vehicle':1})
testFeatures['cp_dose'] = testFeatures['cp_dose'].map({'D1':0, 'D2':1})
testFeatures = testFeatures.drop(columns="sig_id")

featuresCount = trainFeatures.shape[1]
print("Features count = %d" % featuresCount)

targetsCols  = trainTargetScored.columns
targetsCount = len(targetsCols)
print("Targets count = %d" % targetsCount)


def getModel():
	model = Sequential()
	model.add(Dense(1024, input_dim=featuresCount, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(targetsCount, activation="sigmoid"))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[])
	return model

epochs = 20
batchSize = 1000
verbose = 1
nSplits = 11


kfold = KFold(n_splits=nSplits, shuffle=True, random_state=baseSeed)
estimator = KerasClassifier(build_fn=getModel, epochs=epochs, batch_size=batchSize, verbose=verbose)
estimator.fit(trainFeatures, trainTargetScored)

predict = estimator.predict_proba(testFeatures)
predictions = sampleSubmission.copy()
predictions.loc[:,targetsCols] = predict
predictions.to_csv('./databases/lish-moa/submission.csv', index=False)
