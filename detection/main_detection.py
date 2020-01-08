from __future__ import print_function

from pandas import DataFrame
import pandas as pd

from detection.elmo_embed import provideElmo
from preproccesing.load_files import load_glove_and_fastText_model, load_input_data, save_output_data

from preproccesing.preprocesing import clean_messages, tokenize_data, translate_sentence_to_vectors, create_encoders, \
    tokenize_data_reddit, clean_messages_two, translate_sentence_to_vectors_without_save, \
    translate_sentence_to_vectors_without_save_with_elmo
import datetime
import sys

embeddingsPath = 'detection/embeddings/'
input_filesPath = 'input_files/three/'
preprocessed_dataPath = 'detection/preprocessed_data/'
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


class EnvFastText:
    # todo change back
    # model_file = 'wiki-news-300d-1M.vec'
    model_file = 'word2vec_25.txt'

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


env = EnvFastText(data_set=DataSetOne())
# env = EnvFastText(data_set=DataSetReddit())

# todo: zmianiać -> a potem ręcznie połączyć
# env = EnvFastText(data_set=DataSetThreeIrony(), parameter="irony")
# env = EnvFastText(data_set=DataSetThreeSarcasm(), parameter="sarcasm")
# env = EnvFastText(data_set=DataSetThreeRegular(), parameter="regular")

# env = EnvGlove(data_set=DataSetThree())dict(zip(unique, counts))Y_test

# env = EnvGlove(data_set=DataSetOne())
# env = EnvGlove(data_set=DataSetReddit())

dataset_name = env.dataset_name
embedding = env.embedding
model_file = env.model_file
input_file = env.input_file
preprocessed_file = env.preprocessed_file
# todo: change back
# preprocessed_file = 'three_regular_figurative.txt'
preprocessed_file = 'preprocessed_data_fastText_dataset_one.txt'
# preprocessed_file = 'new_new_merged.txt'
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
    save_output_data(data, preprocessed_dataPath + "____new___" + "irony.txt")


def balance_input_data(data: DataFrame) -> DataFrame:
    print('balancing started')
    series = data['Label'].value_counts()
    #todo: coś się jest nie tak z pobiernaiem tego get key, bo to birze chyba po indeksie a nie wartosci
    dataset_count_class_0 = series.get(key=0)
    dataset_count_class_1 = series.get(key=1)
    print(0, 1)
    print(dataset_count_class_0, dataset_count_class_1)
    df_class_0 = data[data['Label'] == 0]
    df_class_1 = data[data['Label'] == 1]

    if (df_class_0.empty or df_class_1.empty):
        df_class_0 = data[data['Label'] == '0']
        df_class_1 = data[data['Label'] == '1']

    if (dataset_count_class_1 > dataset_count_class_0):
        df_class_1_under = df_class_1.sample(dataset_count_class_0)
        df_test_under = pd.concat([df_class_0, df_class_1_under], axis=0)
        data = df_test_under
    elif (dataset_count_class_1 < dataset_count_class_0):
        df_class_0_under = df_class_0.sample(dataset_count_class_1)
        df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
        data = df_test_under

    print('Random under-sampling:')
    print(data['Label'].value_counts())
    data = data.reset_index(drop=True)
    # df_test_under['Label'].value_counts().plot(kind='bar', title='Count (target)');
    print('balancing finished')
    return data


def limit_number_of_data(data: DataFrame, max_number_of_records_per_class) -> DataFrame:
    print('limiting started')
    series = data['Label'].value_counts()
    dataset_count_class_0 = series.get(key=0)
    dataset_count_class_1 = series.get(key=1)
    print(0, 1)
    print(dataset_count_class_0, dataset_count_class_1)
    df_class_0 = data[data['Label'] == 0]
    df_class_1 = data[data['Label'] == 1]

    df_class_0_under = df_class_0.sample(max_number_of_records_per_class)
    df_class_1_under = df_class_1.sample(max_number_of_records_per_class)
    df_test_under = pd.concat([df_class_0_under, df_class_1_under], axis=0)
    data = df_test_under

    print('limiting number of records:')
    print(data['Label'].value_counts())
    data = data.reset_index(drop=True)
    print('limiting finished')
    return data

def reduce_to_max_sentence_length(passed_string: str, max_sentence_length):
    if len(passed_string.split()) > max_sentence_length:
        passed_string = " ".join(passed_string.split()[:50])

    return passed_string

def prepare_data_for_network(max_sentence_length, with_postags, flag='model') -> DataFrame:
    if flag == 'model':
        model = load_glove_and_fastText_model(embeddingsPath + model_file)
        # model = None
    else:
        model = provideElmo()

    # model = None
    print("model loaded")
    data: DataFrame = load_input_data(preprocessed_dataPath + preprocessed_file)
    data['Label'] = data['Label'].astype(int)

    del data['Tweet_index']

    # todo: uncomment
    data = balance_input_data(data)

    data['Tweet_text'] = data['Tweet_text'].apply(reduce_to_max_sentence_length, args=(max_sentence_length,))

    # data = limit_number_of_data(data, 10000)

    print("data loaded")
    if dataset_name == 'reddit':
        tokenize_data_reddit(data)
    else:
        tokenize_data(data)
    print("tokenize_data finished")

    label_encoder, onehot_encoder = create_encoders("detection/")

    # to jest ważne, bez tego nie działa z tf v2.0
    # tf.disable_eager_execution()

    if flag == 'model':
        data: DataFrame = \
            translate_sentence_to_vectors_without_save(data, model,
                                                       output_filename=vector_dataPath + vector_file,
                                                       label_encoder=label_encoder, onehot_encoder=onehot_encoder,
                                                       with_postags=with_postags)
    else:
        data: DataFrame = \
            translate_sentence_to_vectors_without_save_with_elmo(data, model,
                                                                 output_filename=vector_dataPath + vector_file,
                                                                 label_encoder=label_encoder, onehot_encoder=onehot_encoder)

    # print("translate_sentence_to_vectors finished")
    # print("_________________________________")
    # print(list_of_not_found_words)
    # print("size:" + str(len(list_of_not_found_words)))
    return data

# start = datetime.datetime.now()
#
# # debug(input_file)
#
# # debug('model_file')
# # print_all()
# # preprocess_data()
#
# # if dataset_name == 'three':
# #     preprocessed_file = preprocessed_file_clean
# #todo: wywalić hashtag irony z danych, bo nie wywaliłem wczesniej
# prepare_data_for_network()
#
# stop = datetime.datetime.now()
# delta = stop - start
# print(delta)
