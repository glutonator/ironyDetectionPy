import pandas as pd
from gensim.models import KeyedVectors


def load_glove_model():
    # load the Stanford GloVe model
    filename = 'word2vec.txt'
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    return model


def load_input_data():
    # data = pd.read_csv('SemEval2018-T3-train-taskA.txt', sep="\t")
    data = pd.read_csv('test_dane.txt', sep="\t")
    return data
