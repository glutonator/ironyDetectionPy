from typing import Type

import numpy
import numpy as np

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from load_files import load_vectors


def give_data(max_sentence_length):
    data: DataFrame = load_vectors('vector_test_full_good_01_06_2019.txt')

    # # vector
    dataX = data.drop(columns=['Tweet_index', 'Label'])
    # # label
    dataY = data.drop(columns=['Tweet_index', 'Tweet_text'])

    dataX_numpy: Type[numpy.ndarray] = dataX['Tweet_text'].to_numpy(copy=True)
    number_of_sentences = len(dataX_numpy)

    # maksymalna dlugość zdania
    # max_sentence_length = 15
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
    return X_train, X_test, Y_train, Y_test