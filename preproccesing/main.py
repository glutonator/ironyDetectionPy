from __future__ import print_function

from pandas import DataFrame

from preproccesing.load_files import load_glove_and_fastText_model, load_input_data, save_output_data

from preproccesing.preprocesing import clean_messages, tokenize_data, translate_sentence_to_vectors, create_encoders, \
    tokenize_data_reddit, clean_messages_two
import datetime
import sys

embeddingsPath = 'embeddings/'
input_filesPath = 'input_files/three/'
preprocessed_dataPath = 'preprocessed_data/'
vector_dataPath = 'vector_data/'


class ParentDataSet:
    def __init__(self, _dataset_name, _input_file):
        self.dataset_name = _dataset_name
        self.input_file = _input_file


class DataSetOne(ParentDataSet):
    _datasetName = 'one'
    _input_file = 'SemEval2018-T3-train-taskA_merged_with_gold_test_taskA.txt'

    def __init__(self):
        super().__init__(self._datasetName, self._input_file)


class DataSetReddit(ParentDataSet):
    _datasetName = 'reddit'
    _input_file = 'irony-labeled_clean03.csv'

    def __init__(self):
        super().__init__(self._datasetName, self._input_file)


class DataSetThreeSarcasm(ParentDataSet):
    _datasetName = 'three'
    _input_file = 'preprocessed____sarcasm.csv.txt'

    def __init__(self):
        super().__init__(self._datasetName, self._input_file)


class DataSetThreeIrony(ParentDataSet):
    _datasetName = 'three'
    _input_file = 'preprocessed____irony.csv.txt'

    def __init__(self):
        super().__init__(self._datasetName, self._input_file)


class DataSetThreeRegular(ParentDataSet):
    _datasetName = 'three'
    _input_file = 'preprocessed____regular.csv.txt'

    def __init__(self):
        super().__init__(self._datasetName, self._input_file)

class DataSetThreeFigurative(ParentDataSet):
    _datasetName = 'three'
    _input_file = 'preprocessed____figurative.csv.txt'

    def __init__(self):
        super().__init__(self._datasetName, self._input_file)



class EnvFastText:
    model_file = 'wiki-news-300d-1M.vec'

    def __init__(self, data_set: ParentDataSet, parameter: str = ""):
        self.embedding = 'fastText'
        self.dataset_name = data_set.dataset_name
        self.input_file = data_set.input_file
        self.preprocessed_file_clean = 'preprocessed_data_' + self.embedding + '_dataset_' + self.dataset_name + '.txt'
        self.preprocessed_file = 'preprocessed_data_' + self.embedding + '_dataset_' + self.dataset_name + parameter + '.txt'
        self.vector_file = 'vector_data_' + self.embedding + '_dataset_' + self.dataset_name + '.txt'


class EnvGlove:
    model_file = 'word2vec_200.txt'

    def __init__(self, data_set: ParentDataSet):
        self.embedding = 'glove'
        self.dataset_name = data_set.dataset_name
        self.input_file = data_set.input_file
        self.preprocessed_file = 'preprocessed_data_' + self.embedding + '_dataset_' + self.dataset_name + '.txt'
        self.vector_file = 'vector_data_' + self.embedding + '_dataset_' + self.dataset_name + '.txt'


# env = EnvFastText(data_set=DataSetOne())
# env = EnvFastText(data_set=DataSetReddit())

#todo: zmianiać -> a potem ręcznie połączyć
# env = EnvFastText(data_set=DataSetThreeIrony(), parameter="irony")
# env = EnvFastText(data_set=DataSetThreeSarcasm(), parameter="sarcasm")
# env = EnvFastText(data_set=DataSetThreeRegular(), parameter="regular")
env = EnvFastText(data_set=DataSetThreeFigurative(), parameter="figurative")



# env = EnvGlove(data_set=DataSetThree())

# env = EnvGlove(data_set=DataSetOne())
# env = EnvGlove(data_set=DataSetReddit())

dataset_name = env.dataset_name
embedding = env.embedding
model_file = env.model_file
input_file = env.input_file
preprocessed_file = env.preprocessed_file
# preprocessed_file = 'new_merged.txt'
preprocessed_file_clean = env.preprocessed_file_clean
vector_file = env.vector_file


def debug(expression):
    frame = sys._getframe(1)
    print(expression, '=', repr(eval(expression, frame.f_globals, frame.f_locals)))


def print_all():
    print("###############################")
    debug('dataset_name')
    debug('embedding')
    debug('input_file')
    debug('model_file')
    debug('preprocessed_file')
    debug('vector_file')
    print("###############################")


def unify_labels(data: DataFrame):
    # data: DataFrame = load_input_data('ft_preprocessed_data_new3_merged_reddit.txt')
    data['Label'] = data['Label'].replace(-1, 0)


def add_label_based_on_data_file(data: DataFrame, value_to_set):
    data['Label'] = value_to_set


# def preprocess_data():
#     # wczytywanie modelu z plliku:
#     # model = load_glove_and_fastText_model(embeddingsPath + model_file)
#     # data: DataFrame = load_input_data(input_filesPath + input_file)
#     data: DataFrame = load_input_data(preprocessed_dataPath + "tmp/preprocessed_data_fastText_dataset_threeregular.txt")
#     # todo: uncomment
#     # clean_messages(data, model)
#     # clean_messages_two(data)
#
#     data['Label'] = data['Label'].replace(1, 0)
#
#     # replace -1 to 0 in reddit dataset
#     # if dataset_name == 'reddit':
#     #     unify_labels(data)
#
#     # if dataset_name == 'three':
#     #     #todo: zmianiać w zależności od datasetu
#     #     # sarkazm i ironia -> Label =1
#     #     #regular -> label = 0
#     #     # add_label_based_on_data_file(data, 0)
#     #     add_label_based_on_data_file(data, 1)
#
#     # save to file
#     save_output_data(data, preprocessed_dataPath + "____new___"+"regular.txt")


def preprocess_data():
    # wczytywanie modelu z plliku:
    model = load_glove_and_fastText_model(embeddingsPath + model_file)
    data: DataFrame = load_input_data(input_filesPath + input_file)
    #
    # todo: uncomment
    clean_messages(data, model)

    # replace -1 to 0 in reddit dataset
    if dataset_name == 'reddit':
        unify_labels(data)

    if dataset_name == 'three':
        #todo: zmianiać w zależności od datasetu
        # sarkazm i ironia -> Label =1
        #regular -> label = 0
        # add_label_based_on_data_file(data, 0)
        add_label_based_on_data_file(data, 1)

    # save to file
    save_output_data(data, preprocessed_dataPath + preprocessed_file)


def preprocess_data_tmp():
    # wczytywanie modelu z plliku:
    # model = load_glove_and_fastText_model(embeddingsPath + model_file)
    # data: DataFrame = load_input_data(input_filesPath + input_file)
    data: DataFrame = load_input_data(preprocessed_dataPath + "tmp/preprocessed_data_fastText_dataset_threeirony.txt")
    # todo: uncomment
    # clean_messages(data, model)
    clean_messages_two(data)

    # replace -1 to 0 in reddit dataset
    if dataset_name == 'reddit':
        unify_labels(data)

    # if dataset_name == 'three':
    #     #todo: zmianiać w zależności od datasetu
    #     # sarkazm i ironia -> Label =1
    #     #regular -> label = 0
    #     # add_label_based_on_data_file(data, 0)
    #     add_label_based_on_data_file(data, 1)

    # save to file
    save_output_data(data, preprocessed_dataPath + "____new___"+"irony.txt")


def prepare_data_for_network():
    # model = load_glove_model('word2vec_50.txt')
    model = load_glove_and_fastText_model(embeddingsPath + model_file)
    # model = load_FastText_model(embeddingsPath+'word2vec.txt')
    print("model loaded")
    data: DataFrame = load_input_data(preprocessed_dataPath + preprocessed_file)
    del data['Tweet_index']

    print("data loaded")
    if dataset_name == 'reddit':
        tokenize_data_reddit(data)
    else:
        tokenize_data(data)
    print("tokenize_data finished")

    label_encoder, onehot_encoder = create_encoders()

    list_of_not_found_words = \
        translate_sentence_to_vectors(data, model,
                                      output_filename=vector_dataPath + vector_file,
                                      label_encoder=label_encoder, onehot_encoder=onehot_encoder)

    print("translate_sentence_to_vectors finished")
    print("_________________________________")
    print(list_of_not_found_words)
    print("size:" + str(len(list_of_not_found_words)))


start = datetime.datetime.now()

# debug(input_file)

# debug('model_file')
print_all()
preprocess_data()

# if dataset_name == 'three':
#     preprocessed_file = preprocessed_file_clean
#todo: wywalić hashtag irony z danych, bo nie wywaliłem wczesniej
# prepare_data_for_network()

stop = datetime.datetime.now()
delta = stop - start
print(delta)
