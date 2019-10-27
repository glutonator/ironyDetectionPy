from __future__ import print_function

from pandas import DataFrame

from preproccesing.load_files import load_glove_and_fastText_model, load_input_data, save_output_data

from preproccesing.preprocesing import clean_messages, tokenize_data, translate_sentence_to_vectors, create_encoders
import datetime
import sys

embeddingsPath = 'embeddings/'
input_filesPath = 'input_files/'
preprocessed_dataPath = 'preprocessed_data/'
vector_dataPath = 'vector_data/'


class DataSetOne:
    datasetName = 'one'
    input_file = 'SemEval2018-T3-train-taskA_merged_with_gold_test_taskA.txt'


class DataSetReddit:
    datasetName = 'reddit'
    input_file = 'irony-labeled_clean02.csv'


class EnvFastText:
    model_file = 'wiki-news-300d-1M.vec'

    def __init__(self, data_set: DataSetOne):
        self.embedding = 'fastText'
        self.datasetName = data_set.datasetName
        self.input_file = data_set.input_file
        self.preprocessed_file = 'preprocessed_data_' + self.embedding + '_dataset_' + self.datasetName + '.txt'
        self.vector_file = 'vector_data_' + self.embedding + '_dataset_' + self.datasetName + '.txt'


class EnvGlove:
    model_file = 'word2vec.txt'

    def __init__(self, data_set: DataSetOne):
        self.embedding = 'glove'
        self.datasetName = data_set.datasetName
        self.input_file = data_set.input_file
        self.preprocessed_file = 'preprocessed_data_' + self.embedding + '_dataset_' + self.datasetName + '.txt'
        self.vector_file = 'vector_data_' + self.embedding + '_dataset_' + self.datasetName + '.txt'


# env = EnvFastText(data_set=DataSetOne())
env = EnvGlove(data_set=DataSetOne())
# env = EnvFastText(data_set=DataSetReddit())

datasetName = env.datasetName
embedding = env.embedding
model_file = env.model_file
input_file = env.input_file
preprocessed_file = env.preprocessed_file


def debug(expression):
    frame = sys._getframe(1)
    print(expression, '=', repr(eval(expression, frame.f_globals, frame.f_locals)))


def print_all():
    print("###############################")
    debug('datasetName')
    debug('embedding')
    debug('input_file')
    debug('model_file')
    debug('preprocessed_file')
    print("###############################")


def preprocess_data():
    # wczytywanie modelu z plliku:
    model = load_glove_and_fastText_model(embeddingsPath + model_file)
    data: DataFrame = load_input_data(input_filesPath + input_file)
    #
    # todo: uncomment
    # clean_messages(data, model)
    # save to file
    save_output_data(data, preprocessed_dataPath + preprocessed_file)


def prepare_data_for_network():
    # model = load_glove_model('word2vec_50.txt')
    model = load_glove_and_fastText_model(embeddingsPath + model_file)
    # model = load_FastText_model(embeddingsPath+'word2vec.txt')
    print("model loaded")
    data: DataFrame = load_input_data(preprocessed_dataPath + preprocessed_file)
    print("data loaded")
    tokenize_data(data)
    print("tokenize_data finished")

    label_encoder, onehot_encoder = create_encoders()

    list_of_not_found_words = \
        translate_sentence_to_vectors(data, model,
                                      output_filename=vector_dataPath + 'vector_data_fastText_dataset_one.txt',
                                      label_encoder=label_encoder, onehot_encoder=onehot_encoder)

    print("translate_sentence_to_vectors finished")
    print("_________________________________")
    print(list_of_not_found_words)
    print("size:" + str(len(list_of_not_found_words)))


start = datetime.datetime.now()

# debug(input_file)

# debug('model_file')
print_all()
# preprocess_data()
# prepare_data_for_network()

stop = datetime.datetime.now()
delta = stop - start
print(delta)
