import numpy as nplen_of_vector_embeddings
import pandas as pd
from gensim.models import word2vec
from gensim.models import KeyedVectors
import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
import string

from pandas import DataFrame

from load_files import load_glove_model, load_input_data, save_output_data, load_FastText_model
from preprocesing import clean_messages, tokenize_data, translate_sentence_to_vectors, create_encoders
import datetime


def preprocess_data():
    # wczytywanie modelu z plliku:
    model = load_glove_model('word2vec.txt')
    # data: DataFrame = load_input_data('SemEval2018-T3-train-taskA.txt')
    data: DataFrame = load_input_data('SemEval2018-T3-train-taskA_merged_with_gold_test_taskA.txt')
    #
    clean_messages(data, model)
    # save to file
    # save_output_data(data, 'preprocessed_data_new2.txt')
    save_output_data(data, 'preprocessed_data_new3_merged.txt')

def ft_preprocess_data():
    # wczytywanie modelu z plliku:
    model = load_FastText_model('wiki-news-300d-1M.vec')
    # data: DataFrame = load_input_data('SemEval2018-T3-train-taskA.txt')
    data: DataFrame = load_input_data('SemEval2018-T3-train-taskA_merged_with_gold_test_taskA.txt')
    #
    clean_messages(data, model)
    # save to file
    # save_output_data(data, 'preprocessed_data_new2.txt')
    save_output_data(data, 'ft_preprocessed_data_new3_merged.txt')



def prepare_data_for_network():
    # model = load_glove_model('word2vec_50.txt')
    model = load_glove_model('word2vec_200.txt')
    print("model loaded")
    # data: DataFrame = load_input_data('preprocessed_data_new2.txt')
    data: DataFrame = load_input_data('preprocessed_data_new3_merged.txt')
    # data: DataFrame = load_input_data('preprocessed_data_without_blanck_rows.txt')
    # data: DataFrame = load_input_data('preprocessed_data_without_611.txt')
    # data: DataFrame = load_input_data('preprocessed_data.txt')
    # data: DataFrame = load_input_data('xxx.txt')
    print("data loaded")
    tokenize_data(data)
    print("tokenize_data finished")

    label_encoder, onehot_encoder = create_encoders()
    # list_of_not_found_words = \
    #     translate_sentence_to_vectors(data, model, output_filename='vector_test_new_glove_50.txt',
    #                                   label_encoder=label_encoder, onehot_encoder=onehot_encoder)
    list_of_not_found_words = \
        translate_sentence_to_vectors(data, model, output_filename='vector_test_new_glove_merged_200.txt',
                                      label_encoder=label_encoder, onehot_encoder=onehot_encoder)

    print("translate_sentence_to_vectors finished")
    print("_________________________________")
    print(list_of_not_found_words)
    print("size:" + str(len(list_of_not_found_words)))


def ft_prepare_data_for_network():
    # model = load_glove_model('word2vec_50.txt')
    model = load_FastText_model('wiki-news-300d-1M.vec')
    print("model loaded")
    data: DataFrame = load_input_data('ft_preprocessed_data_new3_merged.txt')
    print("data loaded")
    tokenize_data(data)
    print("tokenize_data finished")

    label_encoder, onehot_encoder = create_encoders()
    # list_of_not_found_words = \
    #     translate_sentence_to_vectors(data, model, output_filename='vector_test_new_glove_50.txt',
    #                                   label_encoder=label_encoder, onehot_encoder=onehot_encoder)
    list_of_not_found_words = \
        translate_sentence_to_vectors(data, model, output_filename='vector_test_fast_text_merged.txt',
                                      label_encoder=label_encoder, onehot_encoder=onehot_encoder)

    print("translate_sentence_to_vectors finished")
    print("_________________________________")
    print(list_of_not_found_words)
    print("size:" + str(len(list_of_not_found_words)))




start = datetime.datetime.now()

# preprocess_data()
# ft_preprocess_data()
# prepare_data_for_network()
ft_prepare_data_for_network()


stop = datetime.datetime.now()
delta = stop - start
print(delta)

# print(data['Tweet_text'][0])
# print(data.size)
# print(data.shape[0])
# print()
# print(model.get_vector('sweet'))
#
# TODO: ['we', 'are', 'rumored', 'to', 'have', 'talked', 'to', 'erv', 'agent', '...', 'and', 'the', 'angels', 'asked', 'about', 'ed', 'escobar', '...', "that's", 'hardly', 'nothing', ';)']
# TODO: usunąć wielokropek

# #
#
# # print_all(data)
