from typing import Type

import numpy
import pandas as pd
import numpy as np

from keras import Sequential
from keras.layers import LSTM, Dense, Flatten, AveragePooling1D
from keras_preprocessing.sequence import pad_sequences
from pandas import DataFrame

from load_files import load_vectors
from preprocesing import print_all

import keras
import matplotlib.pyplot as plt

# data: DataFrame = load_vectors('vector_test_20.txt')
# data: DataFrame = load_vectors('vector_test_100.txt')
data: DataFrame = load_vectors('vector_test_full_good_01_06_2019.txt')
print(data.dtypes)
print("___________________")

#
# # vector
dataX = data.drop(columns=['Tweet_index', 'Label'])
# # label
dataY = data.drop(columns=['Tweet_index', 'Tweet_text'])

print(dataX.dtypes)
print(dataX.shape)
print("___________________")

print(dataY.dtypes)
print(dataY.shape)
print("___________________")

print(dataX)

print(dataY)

print("___________________")
# xxxx: np.array = dataX['Tweet_text'].to_numpy(copy=True)
# TODO: tutaj trzeba bedzie poprawić jak bede miał wiecej niz jeden element listy
dataX_numpy: Type[numpy.ndarray] = dataX['Tweet_text'].to_numpy(copy=True)
print(dataX_numpy)
print(type(dataX_numpy))
print("xxxxx")
# print(len(xxxx))
number_of_sentences = len(dataX_numpy)
print(number_of_sentences)

max_sentence_length = 15
list_of_sentences = []
for sentence_index in range(0, number_of_sentences):
    sentence = numpy.asarray(dataX_numpy[sentence_index])

    if len(sentence) < max_sentence_length:
        while len(sentence) < max_sentence_length:
            # print(sentence)
            # print(len(sentence))
            if len(sentence) == 0:
                sentence = np.array([np.zeros(25)])
                # print(sentence)
            zeros = np.array([np.zeros(25)])
            sentence = np.concatenate((sentence, zeros))

        list_of_sentences.append(sentence)
    else:
        sentence = sentence[0:max_sentence_length]
        list_of_sentences.append(sentence)

# print(list_of_sentences)
dataX_numpy = np.asarray(list_of_sentences)
# print(arrayX)
print(dataX_numpy.shape)
# arrayX.reshape((20, 20, 25))
print(dataX_numpy.shape)

print("___________________")
############
##YYYYYYYY
dataY_numpy = numpy.array(dataY['Label'].values)
print(dataY_numpy.shape)

# 25 długoś wektora embediningow
# 11 dlugość zdania (licba wektoróœ
# 1 liczba zdan

len_of_vector_embeddings = 25
model = Sequential()
# model.add(LSTM(32, input_shape=(2, 10), return_sequences=True))
# model.add(LSTM(20, input_shape=(11, 25), return_sequences=False))
model.add(LSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(1))
# model.add(Flatten())
# model.add(Dense(2, activation='softmax'))
# model.add(AveragePooling1D())
# model.add(Flatten())
# model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#

save_best = keras.callbacks.ModelCheckpoint('best.weights', monitor='val_loss', verbose=1, save_best_only=True)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

results = model.fit(dataX_numpy, dataY_numpy, validation_split=0.2,
                    callbacks=[early_stop], epochs=10, batch_size=10,
                    verbose=1)


saveToFile = "robocza_nazwa_loss"
# plt.figure(figsize=[8,6])
plt.plot(results.history['loss'], 'r', linewidth=3.0)
plt.plot(results.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Val Loss'], fontsize=18)
plt.xlabel('Number of epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Plot', fontsize=16)
plt.show()
# plt.savefig(saveToFile)

# print(hist)

# print_all(data)
# print_all(dataX)
# print(dataX)
# print(dataY)

# print(data)
