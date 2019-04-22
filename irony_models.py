import pandas as pd
import numpy as np

from keras import Sequential
from keras.layers import LSTM, Dense
from pandas import DataFrame

from load_files import load_vectors
from preprocesing import print_all

import keras

data: DataFrame = load_vectors()
print(data.dtypes)
print("___________________")

#
# # vector
dataX = data.drop(columns=['Tweet_index', 'Label'])
# # label
dataY = data.drop(columns=['Tweet_index', 'Tweet_text'])

print(dataX.dtypes)
print("___________________")

print(dataY.dtypes)
print("___________________")


print(dataX)

print(dataY)

# print(dataX.to_numpy())
# print(dataY.to_numpy())

model = Sequential()
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(dataX, dataY, epochs=100, batch_size=1, verbose=2)
# model.summary()

# print_all(data)
# print_all(dataX)
# print(dataX)
# print(dataY)

# print(data)
