import re
from typing import List, Any, Union

import nltk
import pandas as pd
import wordninja
from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk import TweetTokenizer
from pandas import DataFrame

from pycontractions import Contractions
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import numpy as np

embeddingsPath = 'embeddings/'


def clean_messages(data: DataFrame, model: Word2VecKeyedVectors):
    # remove urls
    data['Tweet_text'] = data['Tweet_text'].str \
        .replace('(http|ftp|https)://([\\w_-]+(?:(?:\\.[\\w_\\s-]+)+))([\\w.,@?^=%&:/~+#-]*[\\w@?^=%&/~+#-])?', '',
                 regex=True)

    data['Tweet_text'] = data['Tweet_text'].str.replace('\n', ' ', regex=True)


    # remove nicks
    data['Tweet_text'] = data['Tweet_text'].str.replace('@[A-Za-z0-9_]+', ' @ username @ ', regex=True)

    # remove hashtags
    # data['Tweet_text'] = data['Tweet_text'].str.replace('\s([#][\w_-]+)', '', regex=True)
    # remove hastags also from the begging of sentence
    # data['Tweet_text'] = data['Tweet_text'].str.replace('([#][\\w_-]+)', '', regex=True)

    # replace hashtags
    parseHashtags(data)

    # replace time
    data['Tweet_text'] = data['Tweet_text'].str.replace('\\d+:\\d+', ' time ', regex=True)

    # replace number
    data['Tweet_text'] = data['Tweet_text'].str.replace('\\d+', ' number ', regex=True)

    # remove '"'
    data['Tweet_text'] = data['Tweet_text'].str.replace('"', ' ', regex=True)

    # remove “ and ”
    # data['Tweet_text'] = data['Tweet_text'].str.replace('\u201c', ' ', regex=True)
    # data['Tweet_text'] = data['Tweet_text'].str.replace('\u201d', ' ', regex=True)
    #
    #
    # data['Tweet_text'] = data['Tweet_text'].str.replace('\u2019', '\'', regex=True)
    # data['Tweet_text'] = data['Tweet_text'].str.replace('\u2018', '\'', regex=True)



    # remove "|"
    data['Tweet_text'] = data['Tweet_text'].str.replace('|', ' ', regex=False)

    # TODO: dodać tutaj to rozpoznawanie modelu z contractions i konwersje do długich form
    cont = Contractions(kv_model=model)

    number_of_sentences = data.shape[0]
    for i in range(0, number_of_sentences):
        # print(list(cont.expand_texts([data['Tweet_text'][i]]))[-1])
        data['Tweet_text'][i] = list(cont.expand_texts([data['Tweet_text'][i]]))[-1]
        # data['Tweet_text'] = data['Tweet_text'].str.lower()

    print('Contractions finished')

    # convert to lowercase
    data['Tweet_text'] = data['Tweet_text'].str.lower()

    # TODO: dodać lemingi/streming by uciąć 's jeśli bedize z tym problem

    # remove punctuation
    # data['Tweet_text'] = data['Tweet_text'].str.translate(str.maketrans("", ""), str.punctuation)
    # data['Tweet_text'] = data['Tweet_text'].str.translate(string.punctuation)

    # TODO:  KeyError: "word 'erv's' not in vocabulary"
    # data['Tweet_text'] = data['Tweet_text'].str.replace(rf'[{string.punctuation}]', '')


def clean_messages_two(data: DataFrame):
    data['Tweet_text'] = data['Tweet_text'].str.replace('# irony *', ' ', regex=True)
    data['Tweet_text'] = data['Tweet_text'].str.replace('# ironic *', ' ', regex=True)



def parseHashtags(data: DataFrame, ):
    number_of_sentences = data.shape[0]
    for i in range(0, number_of_sentences):
        # print(list(cont.expand_texts([data['Tweet_text'][i]]))[-1])
        sent: str = data['Tweet_text'][i]

        # check if hashtag in sentace
        if bool(re.search('([#][\\w_-]+)', sent)):
            # print(sent)
            sent = sent.split()
            new_sent = []

            for word in sent:
                # print(word)
                isHashtag = bool(re.match('([#][\w_-]+)', word))
                if isHashtag:
                    hashtagWord = wordninja.split(word)
                    # begin string
                    new_sent.append("#")
                    for innerWord in hashtagWord:
                        new_sent.append(innerWord)
                    # end string
                    new_sent.append("*")
                else:
                    new_sent.append(word)

            data['Tweet_text'][i] = " ".join(new_sent)
    print("hashtags Relaced")


def tokenize_data(data):
    tknzr = TweetTokenizer()
    data['Tweet_text'] = data['Tweet_text'].apply(tknzr.tokenize)
    # for i in range(0, data.shape[0]):
    #     print(data['Tweet_text'][i])
    #     data['Tweet_text'][i] = tknzr.tokenize(data['Tweet_text'][i])


def tokenize_data_reddit(data):
    # tknzr = TweetTokenizer()
    # data['Tweet_text'] = data['Tweet_text'].apply(tknzr.tokenize)
    tknzr = TweetTokenizer()
    print(data['Tweet_text'].dtype)
    # col_as_string = data['Tweet_text'].astype('|S')
    col_as_string = data['Tweet_text'].astype(str)
    del data['Tweet_text']
    data['Tweet_text'] = col_as_string
    print(data['Tweet_text'].dtype)
    data['Tweet_text'] = data['Tweet_text'].apply(tknzr.tokenize)


def translate_sentence_to_vectors(data: DataFrame, model: Word2VecKeyedVectors,
                                  output_filename: str, label_encoder: LabelEncoder,
                                  onehot_encoder: OneHotEncoder):
    # silence warnings
    pd.set_option('mode.chained_assignment', None)
    list_of_not_found_words: List[str] = []
    for i in range(0, data.shape[0]):
        list_of_vectors = []

        # add postags
        list_of_tokens_in_sentance = data['Tweet_text'][i]
        list_of_tuples_of_tokens_and_tags = nltk.pos_tag(list_of_tokens_in_sentance)
        list_of_tags_in_order_in_sentance = [i[1] for i in list_of_tuples_of_tokens_and_tags]

        list_of_tags_in_order_in_sentance_in_numeric_labels = label_encoder.transform(list_of_tags_in_order_in_sentance)
        list_of_tags_in_order_in_sentance_in_numeric_labels = list_of_tags_in_order_in_sentance_in_numeric_labels.reshape(len(list_of_tags_in_order_in_sentance_in_numeric_labels), 1)
        list_of_tags_in_order_in_sentance_in_onehot_encoded = onehot_encoder.transform(list_of_tags_in_order_in_sentance_in_numeric_labels)

        count = 0
        # print(data['Tweet_text'][i])
        for word_token in data['Tweet_text'][i]:
            try:
                vector_from_word = model.get_vector(word_token)
                # add postags
                vector_onehot_encoded = list_of_tags_in_order_in_sentance_in_onehot_encoded[count]
                list_of_vectors.append(np.append(vector_from_word, vector_onehot_encoded))
            except KeyError:
                list_of_not_found_words.append(word_token)

            count = count + 1

        data['Tweet_text'][i] = list_of_vectors

    data.to_json(output_filename)
    return list_of_not_found_words


def translate_sentence_to_vectors_without_save(data: DataFrame, model: Word2VecKeyedVectors,
                                  output_filename: str, label_encoder: LabelEncoder,
                                  onehot_encoder: OneHotEncoder) -> DataFrame:
    # silence warnings
    pd.set_option('mode.chained_assignment', None)
    list_of_not_found_words: List[str] = []
    for i in range(0, data.shape[0]):
        list_of_vectors = []

        # add postags
        list_of_tokens_in_sentance = data['Tweet_text'][i]
        list_of_tuples_of_tokens_and_tags = nltk.pos_tag(list_of_tokens_in_sentance)
        list_of_tags_in_order_in_sentance = [i[1] for i in list_of_tuples_of_tokens_and_tags]

        list_of_tags_in_order_in_sentance_in_numeric_labels = label_encoder.transform(list_of_tags_in_order_in_sentance)
        list_of_tags_in_order_in_sentance_in_numeric_labels = list_of_tags_in_order_in_sentance_in_numeric_labels.reshape(len(list_of_tags_in_order_in_sentance_in_numeric_labels), 1)
        list_of_tags_in_order_in_sentance_in_onehot_encoded = onehot_encoder.transform(list_of_tags_in_order_in_sentance_in_numeric_labels)

        count = 0
        # print(data['Tweet_text'][i])
        for word_token in data['Tweet_text'][i]:
            try:
                vector_from_word = model.get_vector(word_token)
                # add postags
                vector_onehot_encoded = list_of_tags_in_order_in_sentance_in_onehot_encoded[count]
                list_of_vectors.append(np.append(vector_from_word, vector_onehot_encoded))
            except KeyError:
                list_of_not_found_words.append(word_token)

            count = count + 1

        data['Tweet_text'][i] = list_of_vectors

    return data
    # data.to_json(output_filename)
    # return list_of_not_found_words



def print_all(data):
    print(list(data.columns.values))
    print(data['Tweet_index'].values)
    print(data['Label'].values)
    print(data['Tweet_text'].values)
    print(data.dtypes)


def create_encoders(parentDir : str = ''):
    list_of_pos_tags = []
    with open(parentDir+embeddingsPath+'pos_tags.txt', 'r') as f:
        for line in f:
            line = line.rstrip()
            print(line)
            list_of_pos_tags.append(line)
            # Do something with 'line'

    print(list_of_pos_tags)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(list_of_pos_tags)
    print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return label_encoder, onehot_encoder

