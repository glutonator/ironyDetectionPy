import pandas as pd
import numpy as np

from keras import Sequential
from keras.layers import LSTM, Dense, Flatten, AveragePooling1D
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
print(dataX.shape)
# print(dataX['Tweet_text'].to_numpy())
print("___________________")

print(dataY.dtypes)
print(dataY.shape)
print("___________________")

# print(dataX)
#
# print(dataY)

# print(dataX.to_numpy())
# print(dataY.to_numpy())

model = Sequential()
# model.add(LSTM(32, input_shape=(2, 10), return_sequences=True))
model.add(LSTM(1, input_shape=(20, 1), return_sequences=False))
# model.add(Dense(1))
# model.add(Flatten())
# model.add(Dense(2, activation='softmax'))
# model.add(AveragePooling1D())
# model.add(Flatten())
# model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam')
#

xxxx = np.arange(20)
xxxx = xxxx.reshape((1, 20, 1))
print(xxxx.shape)
yyyy = np.arange(1)
# yyyy.reshape(20, 1, 1)
print(yyyy.shape)
print(yyyy)

model.fit(xxxx, yyyy, epochs=100, batch_size=1, verbose=2)

# model.summary()

# print_all(data)
# print_all(dataX)
# print(dataX)
# print(dataY)

# print(data)
