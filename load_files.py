import pandas as pd
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors


def load_glove_model() -> Word2VecKeyedVectors:
    # load the Stanford GloVe model
    filename = 'word2vec.txt'
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    return model


def load_input_data() -> pd.DataFrame:
    # data = pd.read_csv('SemEval2018-T3-train-taskA.txt', sep="\t")
    data = pd.read_csv('test_dane.txt', sep="\t")
    return data


def load_vectors() -> pd.DataFrame:
    # data = pd.read_csv('SemEval2018-T3-train-taskA.txt', sep="\t")
    # data = pd.read_csv('vector_test.txt', converters={'Tweet_text' : pd.eval})
    data = pd.read_json('vector_test.txt')
    data=data.sort_index()
    return data