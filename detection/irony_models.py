import datetime
from typing import Type

import keras
import matplotlib.pyplot as plt
import numpy
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from load_files import load_vectors

data: DataFrame = load_vectors('vector_test_full_good_01_06_2019.txt')

# # vector
dataX = data.drop(columns=['Tweet_index', 'Label'])
# # label
dataY = data.drop(columns=['Tweet_index', 'Tweet_text'])

dataX_numpy: Type[numpy.ndarray] = dataX['Tweet_text'].to_numpy(copy=True)
number_of_sentences = len(dataX_numpy)

# maksymalna dlugość zdania
max_sentence_length = 15
list_of_sentences = []
for sentence_index in range(0, number_of_sentences):
    sentence = numpy.asarray(dataX_numpy[sentence_index])

    if len(sentence) < max_sentence_length:
        while len(sentence) < max_sentence_length:
            if len(sentence) == 0:
                sentence = np.array([np.zeros(25)])
            zeros = np.array([np.zeros(25)])
            sentence = np.concatenate((sentence, zeros))

        list_of_sentences.append(sentence)
    else:
        sentence = sentence[0:max_sentence_length]
        list_of_sentences.append(sentence)

dataX_numpy = np.asarray(list_of_sentences)
dataY_numpy = numpy.array(dataY['Label'].values)
print(dataY_numpy.shape)

# split training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(dataX_numpy, dataY_numpy, test_size=0.2, random_state=42)
print("data split for training and testing")


#########################
# model
# dlugosc wektora embedingow
len_of_vector_embeddings = 25
model = Sequential()
# model.add(LSTM(32, input_shape=(2, 10), return_sequences=True))
# model.add(LSTM(20, input_shape=(11, 25), return_sequences=False))
# model.add(LSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=False))
# model.add(LSTM(50, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=True))
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(max_sentence_length, len_of_vector_embeddings)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(20, return_sequences=False)))
# model.add(LSTM(20, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#

save_best = keras.callbacks.ModelCheckpoint('weights.best.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

results = model.fit(X_train, Y_train, validation_split=0.2,
                    callbacks=[early_stop, save_best], epochs=30, batch_size=10,
                    verbose=0)

time = datetime.datetime.now().time()
print(results.history.keys())
saveToFile = "robocza_nazwa_loss"
plt.plot(results.history['loss'], 'r', linewidth=3.0)
plt.plot(results.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Val Loss'], fontsize=18)
plt.xlabel('Number of epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Plot' + str(time), fontsize=16)
plt.show()
plt.clf()
# plt.savefig(saveToFile)

saveToFile = "robocza_nazwa_accuracy"
plt.plot(results.history['acc'], 'r', linewidth=3.0)
plt.plot(results.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Val Accuracy'], fontsize=18)
plt.xlabel('Number of epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Plot' + str(time), fontsize=16)
plt.show()
plt.clf()
# plt.savefig("wykresy/" + saveToFile)

scores = model.evaluate(X_test, Y_test, verbose=1)
print(model.metrics_names)
print(scores)

