import numpy as np
import pandas as pd
import pickle

from keras.models import load_model
from sklearn.decomposition import PCA
from tensorflow import random

random.set_seed(5577)

# Loading the model and PCA fit
model = load_model('../input/jsmp-dnn-training-keras/model.h5')
model.summary()

unpickle = open("../input/jsmp-dnn-training-keras/pcaFit.pkl", 'rb')
pca = pickle.load(unpickle)

# Submission testing
import janestreet
env = janestreet.make_env()
iter_test = env.iter_test()
counter = 0
for (test_df, sample_prediction_df) in iter_test:
    counter += 1
    test_df.fillna(0, inplace=True)

    if test_df['weight'].values == 0:
        predDf = 0
    else:
        predPCA = pca.transform(test_df)
        prediction = model.predict(predPCA)
        if prediction > 0.5:
            predDf = 1
        else:
            predDf = 0
    sample_prediction_df.action = predDf
    env.predict(sample_prediction_df)

print("Total prediction : %d" % counter)
print("Submission done!")
