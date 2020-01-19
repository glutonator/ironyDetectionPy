from typing import Type

import numpy
import numpy as np

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from preproccesing.load_files import load_vectors
from collections import Counter

main_dir = 'detection/'
vector_dataPath = main_dir + 'vector_data/'


def get_data_from_dataset_one(len_of_vector_embeddings, max_sentence_length):
    # todo: split glove and fasttext
    data: DataFrame = load_vectors(vector_dataPath + 'vector_data_fastText_dataset_one.txt')
    get_balance_info(data, "one")

    # # vector
    dataX = data.drop(columns=['Tweet_index', 'Label'])
    # # label
    dataY = data.drop(columns=['Tweet_index', 'Tweet_text'])

    dataX_numpy: Type[numpy.ndarray] = dataX['Tweet_text'].to_numpy(copy=True)
    number_of_sentences = len(dataX_numpy)

    count_of_list_of_sentences = []

    # maksymalna dlugość zdania
    # max_sentence_length = 15
    max_len = 0
    list_of_sentences = []
    for sentence_index in range(0, number_of_sentences):
        sentence = numpy.asarray(dataX_numpy[sentence_index])
        # if max_len < len(sentence):
        #     max_len = len(sentence)
        # print(str(sentence_index) + " " + str(max_len))
        # if(len(sentence)>20):
        #     print(str(sentence_index) + " " + str(len(sentence)))
        count_of_list_of_sentences.append("" + str(len(sentence)))

        if len(sentence) < max_sentence_length:
            while len(sentence) < max_sentence_length:
                if len(sentence) == 0:
                    sentence = np.array([np.zeros(len_of_vector_embeddings)])
                zeros = np.array([np.zeros(len_of_vector_embeddings)])
                sentence = np.concatenate((sentence, zeros))

            list_of_sentences.append(sentence)
        else:
            sentence = sentence[0:max_sentence_length]
            list_of_sentences.append(sentence)

    print("##############")
    print("count_of_list_of_sentences")
    print(Counter(count_of_list_of_sentences))

    dataX_numpy = np.asarray(list_of_sentences)
    dataY_numpy = numpy.array(dataY['Label'].values)
    print(dataY_numpy.shape)

    return dataX_numpy, dataY_numpy


def get_data_from_dataset_three(data: DataFrame, len_of_vector_embeddings, max_sentence_length):
    # todo: split glove and fasttext
    # data: DataFrame = load_vectors(vector_dataPath + 'vector_data_fastText_dataset_three.txt')
    # data: DataFrame = load_vectors(vector_dataPath + 'vector_data_fastText_dataset_one.txt')
    get_balance_info(data, "three")
    print("XXXXXXXXXXXXXXX")
    print("data loaded from file")

    # # vector
    dataX = data.drop(columns=['Label'])
    # # label
    dataY = data.drop(columns=['Tweet_text'])

    dataX_numpy: Type[numpy.ndarray] = dataX['Tweet_text'].to_numpy(copy=True)
    number_of_sentences = len(dataX_numpy)

    other_count = dataX['Tweet_text'].str.len()

    count_of_list_of_sentences = []

    # maksymalna dlugość zdania
    # max_sentence_length = 15
    max_len = 0
    list_of_sentences = []
    for sentence_index in range(0, number_of_sentences):
        sentence = numpy.asarray(dataX_numpy[sentence_index])
        # if max_len < len(sentence):
        #     max_len = len(sentence)
        # print(str(sentence_index) + " " + str(max_len))
        # if(len(sentence)>20):
        #     print(str(sentence_index) + " " + str(len(sentence)))
        count_of_list_of_sentences.append("" + str(len(sentence)))

        if len(sentence) < max_sentence_length:
            while len(sentence) < max_sentence_length:
                if len(sentence) == 0:
                    sentence = np.array([np.zeros(len_of_vector_embeddings)])
                zeros = np.array([np.zeros(len_of_vector_embeddings)])
                sentence = np.concatenate((sentence, zeros))

            list_of_sentences.append(sentence)
        else:
            sentence = sentence[0:max_sentence_length]
            list_of_sentences.append(sentence)

    print("##############")
    print("count_of_list_of_sentences")
    print(Counter(count_of_list_of_sentences))

    dataX_numpy = np.asarray(list_of_sentences)
    dataY_numpy = numpy.array(dataY['Label'].values)
    print(dataY_numpy.shape)

    return dataX_numpy, dataY_numpy


def get_data_from_dataset_reddit(len_of_vector_embeddings, max_sentence_length):
    # todo: split glove and fasttext
    data: DataFrame = load_vectors(vector_dataPath + 'vector_data_fastText_dataset_reddit.txt')
    get_balance_info(data, "reddit")

    # # vector
    dataX = data.drop(columns=['Label'])
    # # label
    dataY = data.drop(columns=['Tweet_text'])

    dataX_numpy: Type[numpy.ndarray] = dataX['Tweet_text'].to_numpy(copy=True)
    number_of_sentences = len(dataX_numpy)

    count_of_list_of_sentences = []

    # maksymalna dlugość zdania
    # max_sentence_length = 15
    max_len = 0
    list_of_sentences = []
    for sentence_index in range(0, number_of_sentences):
        sentence = numpy.asarray(dataX_numpy[sentence_index])
        # if max_len < len(sentence):
        #     max_len = len(sentence)
        # print(str(sentence_index) + " " + str(max_len))
        # if(len(sentence)>20):
        #     print(str(sentence_index) + " " + str(len(sentence)))
        count_of_list_of_sentences.append("" + str(len(sentence)))

        if len(sentence) < max_sentence_length:
            while len(sentence) < max_sentence_length:
                if len(sentence) == 0:
                    sentence = np.array([np.zeros(len_of_vector_embeddings)])
                zeros = np.array([np.zeros(len_of_vector_embeddings)])
                sentence = np.concatenate((sentence, zeros))

            list_of_sentences.append(sentence)
        else:
            sentence = sentence[0:max_sentence_length]
            list_of_sentences.append(sentence)

    print("##############")
    print("count_of_list_of_sentences")
    print(Counter(count_of_list_of_sentences))

    dataX_numpy = np.asarray(list_of_sentences)
    dataY_numpy = numpy.array(dataY['Label'].values)
    print(dataY_numpy.shape)

    return dataX_numpy, dataY_numpy


def get_data_for_network(total_length, max_sentence_length, sets):
    if sets == 'both':
        XXXX_one, YYYY_one = get_data_from_dataset_one(total_length, max_sentence_length)
        XXXX_reddit, YYYY_reddit = get_data_from_dataset_reddit(total_length, max_sentence_length)

        XXXX = np.append(XXXX_one, XXXX_reddit, axis=0)
        YYYY = np.append(YYYY_one, YYYY_reddit, axis=0)
        return XXXX, YYYY

    elif sets == 'one':
        XXXX_one, YYYY_one = get_data_from_dataset_one(total_length, max_sentence_length)
        return XXXX_one, YYYY_one

    elif sets == 'red':
        XXXX_reddit, YYYY_reddit = get_data_from_dataset_reddit(total_length, max_sentence_length)
        return XXXX_reddit, YYYY_reddit

    elif sets == 'three':
        XXXX_one, YYYY_one = get_data_from_dataset_three(total_length, max_sentence_length)
        return XXXX_one, YYYY_one


def get_balance_info(data, dataset_name):
    dataset_count = data['Label'].value_counts()
    print(dataset_name + ": \n" + str(dataset_count))

def get_balance_info_from_array(array, dataset_name):
    # www = np.array([1, 1, 1, 0, 0])
    unique, counts = numpy.unique(array, return_counts=True)
    print(dataset_name + ": \n" + str(dict(zip(unique, counts))))
    # dataset_count = data['Label'].value_counts()
    # print(dataset_name + ": \n" + str(dataset_count))


def split_data_sets(dataX_numpy, dataY_numpy, XXXX_test= None, YYYY_test= None):

    if XXXX_test is None:
        # split training and testing data
        # todo zmianić test_size=0.9 na 0.2
        X_train_2, X_test, Y_train_2, Y_test = train_test_split(dataX_numpy, dataY_numpy, test_size=0.2, random_state=42)
        # X_train_2, X_test, Y_train_2, Y_test = train_test_split(dataX_numpy, dataY_numpy, test_size=0.2, random_state=None)

        X_train, X_val, Y_train, Y_val = train_test_split(X_train_2, Y_train_2, test_size=0.2, random_state=42)
        # X_train, X_val, Y_train, Y_val = train_test_split(X_train_2, Y_train_2, test_size=0.2, random_state=None)
    else:
        X_train, X_val, Y_train, Y_val = train_test_split(dataX_numpy, dataY_numpy, test_size=0.2, random_state=42)
        # X_train, X_val, Y_train, Y_val = train_test_split(dataX_numpy, dataY_numpy, test_size=0.2, random_state=None)
        X_test = XXXX_test
        Y_test = YYYY_test


    get_balance_info_from_array(Y_train,"Y_train")
    get_balance_info_from_array(Y_val,"Y_val")
    get_balance_info_from_array(Y_test,"Y_test")

    print("data split for training and testing")
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
