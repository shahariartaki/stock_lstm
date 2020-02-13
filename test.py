import pandas as pd

from matplotlib import pyplot as plt
import numpy as np

pd.__version__
df = pd.read_csv("test_1.csv")
training_set = df.iloc[:, 1:2].values
training_set1 = df.iloc[:, 1:2].values
#tk=(df[df['TRADING CODE'] == 'GREENDELT'])
#print(df)

plt.figure()
plt.plot(df["CLOSEP"])
plt.title('GREENDELT stock price history')
plt.ylabel('Price (BDT)')
plt.xlabel('DATE')
plt.legend(['OPENP','HIGH','LOW','CLOSEP'], loc='upper left')
plt.show()

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
print(training_set_scaled)
sc1 = MinMaxScaler(feature_range = (0, 1))
training_set_scaled1 = sc1.fit_transform(training_set1)

print(training_set_scaled1)


plt.figure()
plt.plot(training_set_scaled)
plt.title('GREENDELT stock price history')
plt.ylabel('Price (BDT)')
plt.xlabel('DATE')
plt.legend(['OPENP','HIGH','LOW','CLOSEP'], loc='upper left')
plt.show()
