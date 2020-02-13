# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:32:20 2019

@author: Taki
"""
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

pd.__version__
df = pd.read_csv("data.csv")
tk=(df[df['TRADING CODE'] == 'GREENDELT'])
print(tk)

plt.figure()
plt.plot(tk["OPENP"])
plt.plot(tk["HIGH"])
plt.plot(tk["LOW"])
plt.plot(tk["CLOSEP"])
plt.title('GREENDELT stock price history')
plt.ylabel('Price (BDT)')
plt.xlabel('DATE')
plt.legend(['OPENP','HIGH','LOW','CLOSEP'], loc='upper left')
plt.show()

data_norm_to_0_1 = [number/scipy.linalg.norm(data) for number in data]

trace1 = go.Histogram(
    x=data,
    opacity=0.75,
    name='data',
)

trace2 = go.Histogram(
    x=data_norm_to_0_1,
    opacity=0.75,
    name='normalized to [0,1]',
)

fig = tools.make_subplots(rows=2, cols=1)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)

fig['layout'].update(height=600, width=800, title='Normalize to [0,1]')
py.iplot(fig, filename='apple-data-normalize-0-1')



plt.figure()
plt.plot(tk["VOLUME"])
plt.title('GREENDELT stock volume history')
plt.ylabel('VOLUME')
plt.xlabel('DATE')
plt.show()

#plt.figure()
#plt.plot(tk["DATE"])
#plt.title('GREENDELT stock volume history')
#plt.ylabel('DATE')
#plt.show()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

train_cols = ["OPENP","HIGH","LOW","CLOSEP","VOLUME"]
df_train, df_test = train_test_split(tk, train_size=0.8, test_size=0.2, shuffle=False)
print("Train and Test size", len(df_train), len(df_test))
# scale the feature MinMax, build array
x = df_train.loc[:,train_cols].values
y = df_train.loc[:,train_cols].values
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])


y_col_index=1059
def build_timeseries(mat, y_col_index):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0, TIME_STEPS, dim_1))
    
    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y



scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(x)
rescaledY = scaler.fit_transform(y)
# summarize transformed data
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])
plt.figure()
plt.plot(rescaledX[0:5,:])
plt.title('nomalaization')
plt.show()


print(rescaledY[0:5,:])
plt.figure()
plt.plot(rescaledY[0:5,:])
plt.title('nomalaization 2')
plt.ylabel(rescaledY)
plt.xlabel(rescaledX)
plt.show()
