import gc

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

from tensorflow import random

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sn


random.set_seed(5577)

# Data Loading
trainDf = pd.read_csv('/kaggle/input/jane-street-market-prediction/train.csv')

# Reducing Memory Usage
def reduce_memory_usage(df):
    start_memory = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_memory} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    pass
        else:
            df[col] = df[col].astype('category')

    end_memory = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe after reduction {end_memory} MB")
    print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")
    return df

trainDf = reduce_memory_usage(trainDf)

# Data Preparation
dropCols = ["resp", "resp_1", "resp_2", "resp_3", "resp_4", "ts_id"]
trainDf = trainDf.drop(columns=dropCols)
print(trainDf.isnull().sum())
trainDf.fillna(0, inplace=True)
trainDfW = trainDf[trainDf["weight"] > 0]

# Basic Data Exploration
print(trainDf.shape)
print(trainDfW.shape)
print(trainDfW.head())
print(trainDfW.describe())

# Data Comprehension
## Correlation Matrix
corrDfW = trainDfW.corr()
fig, ax = plt.subplots(figsize=(25,25))
sn.heatmap(corrDfW, linewidths=.5, annot=False, ax=ax)
plt.show()

## Generic PCA
scaler = MinMaxScaler()
scaledTrain = scaler.fit_transform(trainDfW)

pca = PCA().fit(scaledTrain)
exCumul = np.cumsum(pca.explained_variance_ratio_)
px.area(
    x=range(1, exCumul.shape[0] + 1),
    y=exCumul,
    labels={"x": "# Components", "y": "Explained Variance"}
)

## 2D PCA - Allow to observe that feature_0 discriminate the data point into two distinct clusters
pca = PCA(n_components=2)
dfComp = pca.fit_transform(scaledTrain)

total_var = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter(dfComp, x=0, y=1, color=trainDfW['weight'], title=f'Total Explained Variance: {total_var:.3f}%', labels={'0': 'PC 1', '1': 'PC 2'})
fig.show()

## 2D PCA without feature_0 - data points are filling a nicely shaped rhomboid
dfNoF0 = trainDfW.drop("feature_0", 1)
scaledTrainNoF0 = scaler.fit_transform(dfNoF0)
pca = PCA(n_components=2)
dfComp = pca.fit_transform(scaledTrainNoF0)

total_var = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter(dfComp, x=0, y=1, color=trainDfW['weight'], title=f'Total Explained Variance: {total_var:.3f}%', labels={'0': 'PC 1', '1': 'PC 2'})
fig.show()
