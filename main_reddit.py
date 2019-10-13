import numpy as nplen_of_vector_embeddings
import pandas as pd
from gensim.models import word2vec
from gensim.models import KeyedVectors
import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
import string

from pandas import DataFrame

from load_files import load_glove_model, load_input_data, save_output_data, load_FastText_model, load_input_data_reddit
from preprocesing import clean_messages, tokenize_data, translate_sentence_to_vectors, create_encoders, \
    tokenize_data_reddit
import datetime


# def preprocess_data():
#     # wczytywanie modelu z plliku:
#     model = load_glove_model('word2vec.txt')
#     # data: DataFrame = load_input_data('SemEval2018-T3-train-taskA.txt')
#     data: DataFrame = load_input_data('SemEval2018-T3-train-taskA_merged_with_gold_test_taskA.txt')
#     #
#     clean_messages(data, model)
#     # save to file
#     # save_output_data(data, 'preprocessed_data_new2.txt')
#     save_output_data(data, 'preprocessed_data_new3_merged.txt')
#
# def ft_preprocess_data():
#     # wczytywanie modelu z plliku:
#     model = load_FastText_model('wiki-news-300d-1M.vec')
#     # data: DataFrame = load_input_data('SemEval2018-T3-train-taskA.txt')
#     data: DataFrame = load_input_data('SemEval2018-T3-train-taskA_merged_with_gold_test_taskA.txt')
#     #
#     clean_messages(data, model)
#     # save to file
#     # save_output_data(data, 'preprocessed_data_new2.txt')
#     save_output_data(data, 'ft_preprocessed_data_new3_merged.txt')

def ft_preprocess_data_reddit():
    # wczytywanie modelu z plliku:
    # model = load_glove_model('word2vec.txt')
    model = load_FastText_model('wiki-news-300d-1M.vec')
    # data: DataFrame = load_input_data('SemEval2018-T3-train-taskA.txt')
    # data: DataFrame = load_input_data_reddit('irony-labeled_test.csv')
    # data: DataFrame = load_input_data_reddit('irony-labeled_test.csv')
    # data: DataFrame = load_input_data_reddit('irony-labeled_clean.csv')
    data: DataFrame = load_input_data_reddit('irony-labeled_clean02.csv')
    #
    unifyLabels(data)
    clean_messages(data, model)
    # save to file
    # save_output_data(data, 'preprocessed_data_new2.txt')
    save_output_data(data, 'ft_preprocessed_data_new3_merged_reddit.txt')
    # save_output_data(data, 'ft_preprocessed_data_new3_merged_reddit_test.txt')





# def prepare_data_for_network():
#     # model = load_glove_model('word2vec_50.txt')
#     model = load_glove_model('word2vec_200.txt')
#     print("model loaded")
#     # data: DataFrame = load_input_data('preprocessed_data_new2.txt')
#     data: DataFrame = load_input_data('preprocessed_data_new3_merged.txt')
#     # data: DataFrame = load_input_data('preprocessed_data_without_blanck_rows.txt')
#     # data: DataFrame = load_input_data('preprocessed_data_without_611.txt')
#     # data: DataFrame = load_input_data('preprocessed_data.txt')
#     # data: DataFrame = load_input_data('xxx.txt')
#     print("data loaded")
#     tokenize_data(data)
#     print("tokenize_data finished")
#
#     label_encoder, onehot_encoder = create_encoders()
#     # list_of_not_found_words = \
#     #     translate_sentence_to_vectors(data, model, output_filename='vector_test_new_glove_50.txt',
#     #                                   label_encoder=label_encoder, onehot_encoder=onehot_encoder)
#     list_of_not_found_words = \
#         translate_sentence_to_vectors(data, model, output_filename='vector_test_new_glove_merged_200.txt',
#                                       label_encoder=label_encoder, onehot_encoder=onehot_encoder)
#
#     print("translate_sentence_to_vectors finished")
#     print("_________________________________")
#     print(list_of_not_found_words)
#     print("size:" + str(len(list_of_not_found_words)))


def ft_prepare_data_for_network_reddit():
    # model = load_glove_model('word2vec.txt')
    # model = load_glove_model('word2vec_50.txt')
    model = load_FastText_model('wiki-news-300d-1M.vec')
    print("model loaded")
    data: DataFrame = load_input_data('ft_preprocessed_data_new3_merged_reddit.txt')
    # data: DataFrame = load_input_data('ft_preprocessed_data_new3_merged_reddit_test.txt')
    # unifyLabels(data)
    print("data loaded")
    tokenize_data_reddit(data)
    # tokenize_data(data)
    print("tokenize_data finished")

    label_encoder, onehot_encoder = create_encoders()
    # list_of_not_found_words = \
    #     translate_sentence_to_vectors(data, model, output_filename='vector_test_new_glove_50.txt',
    #                                   label_encoder=label_encoder, onehot_encoder=onehot_encoder)
    list_of_not_found_words = \
        translate_sentence_to_vectors(data, model, output_filename='vector_test_fast_text_merged_reddit.txt',
                                      label_encoder=label_encoder, onehot_encoder=onehot_encoder)

    print("translate_sentence_to_vectors finished")
    print("_________________________________")
    print(list_of_not_found_words)
    print("size:" + str(len(list_of_not_found_words)))


def unifyLabels(data: DataFrame):
    # data: DataFrame = load_input_data('ft_preprocessed_data_new3_merged_reddit.txt')
    data['Label'] = data['Label'].replace(-1, 0)


start = datetime.datetime.now()

# ft_preprocess_data_reddit()
ft_prepare_data_for_network_reddit()



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
