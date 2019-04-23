from typing import Type

import numpy
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

print(dataX)

print(dataY)

print("___________________")
# xxxx: np.array = dataX['Tweet_text'].to_numpy(copy=True)
# TODO: tutaj trzeba bedzie poprawić jak bede miał wiecej niz jeden element listy
xxxx: Type[numpy.ndarray] = dataX['Tweet_text'].to_numpy(copy=True)
# print(xxxx)
# print(type(xxxx))
XXXX = numpy.asarray(xxxx[0])
# print(XXXX)
# print(type(XXXX))
# print(XXXX.shape)
XXXX = XXXX.reshape((1, 11, 25))

# print(XXXX.shape)

print("___________________")
############
##YYYYYYYY
YYYY = numpy.array(dataY['Label'].values)
print(YYYY.shape)

for sentence in XXXX:
    print(len(sentence))
    for word in sentence:
        print(len(word))

# print(str(XXXX))
# print(len(XXXX))

# print(dataX.to_numpy())
# print(dataY.to_numpy())

model = Sequential()
# model.add(LSTM(32, input_shape=(2, 10), return_sequences=True))
model.add(LSTM(20, input_shape=(11, 25), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(1))
# model.add(Flatten())
# model.add(Dense(2, activation='softmax'))
# model.add(AveragePooling1D())
# model.add(Flatten())
# model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam')
#

# xxxx = np.arange(20)
# xxxx = xxxx.reshape((1, 20, 1))
# print(xxxx.shape)
# yyyy = np.arange(1)
# # yyyy.reshape(20, 1, 1)
# print(yyyy.shape)
# print(yyyy)


model.fit(XXXX, YYYY, epochs=2, batch_size=1, verbose=2)

# model.summary()

# print_all(data)
# print_all(dataX)
# print(dataX)
# print(dataY)

# print(data)
