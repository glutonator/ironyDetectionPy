import re
from collections import Counter
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

from detection.elmo_embed import elmo_vectors, divide_chunks

embeddingsPath = 'embeddings/'


#todo: check if emoji exist in vord to vec
def split_emoji(passed_string: str):
    matches: list[str] = re.findall(r"[:][\w-]+[:]", passed_string)
    for match in matches:
        replaced_string: str = match.replace(':', '').replace('_', " ").replace('-', " ")
        replaced_string = ' emote ' + replaced_string
        passed_string = passed_string.replace(match, replaced_string)

    return passed_string

#(([[a-z]+[_])+)
# :(([[a-z]+[_])+[a-z]+):

def clean_messages222(data: DataFrame, model: Word2VecKeyedVectors):
    data['Tweet_text'] = data['Tweet_text'].apply(split_emoji)


def clean_messages(data: DataFrame, model: Word2VecKeyedVectors):
    # remove urls
    data['Tweet_text'] = data['Tweet_text'].str \
        .replace('(http|ftp|https)://([\\w_-]+(?:(?:\\.[\\w_\\s-]+)+))([\\w.,@?^=%&:/~+#-]*[\\w@?^=%&/~+#-])?', ' url ',
                 regex=True)

    data['Tweet_text'] = data['Tweet_text'].str.replace('\n', ' ', regex=True)

    # split emoji
    data['Tweet_text'] = data['Tweet_text'].apply(split_emoji)

    # remove nicks
    data['Tweet_text'] = data['Tweet_text'].str.replace('@[A-Za-z0-9_]+', ' @ username @ ', regex=True)

    # replace hashtags todo: uncomment !!!!!!!!!!!!1
    parseHashtags(data)

    # replace time
    data['Tweet_text'] = data['Tweet_text'].str.replace('\\d+:\\d+', ' time ', regex=True)

    # replace number
    data['Tweet_text'] = data['Tweet_text'].str.replace('\\d+', ' number ', regex=True)

    # remove '"'
    data['Tweet_text'] = data['Tweet_text'].str.replace('"', ' ', regex=True)

    # remove "|"
    data['Tweet_text'] = data['Tweet_text'].str.replace('|', ' ', regex=False)

    # remove white dots combos
    data['Tweet_text'] = data['Tweet_text'].str.replace('\\s\\.\\s\\.\\s', '.', regex=True)
    data['Tweet_text'] = data['Tweet_text'].str.replace('\\.\\s\\.', '.', regex=True)
    data['Tweet_text'] = data['Tweet_text'].str.replace('\\.+', '.', regex=True)
    # data['Tweet_text'] = data['Tweet_text'].apply(lambda x: " ".join(x.split()))

    # " ".join(s.split())

    # remove "..."
    data['Tweet_text'] = data['Tweet_text'].str.replace('...', '.', regex=False)

    # remove ".."
    data['Tweet_text'] = data['Tweet_text'].str.replace('..', '.', regex=False)


    # todo: uncomment if needed comented clearing contractions
    # process_contractions(data, model)

    # convert to lowercase
    data['Tweet_text'] = data['Tweet_text'].str.lower()

    # TODO: dodać lemingi/streming by uciąć 's jeśli bedize z tym problem

    # remove punctuation
    # data['Tweet_text'] = data['Tweet_text'].str.translate(str.maketrans("", ""), str.punctuation)
    # data['Tweet_text'] = data['Tweet_text'].str.translate(string.punctuation)

    # TODO:  KeyError: "word 'erv's' not in vocabulary"
    # data['Tweet_text'] = data['Tweet_text'].str.replace(rf'[{string.punctuation}]', '')


def process_contractions(data, model):
    # TODO: dodać tutaj to rozpoznawanie modelu z contractions i konwersje do długich form
    cont = Contractions(kv_model=model)
    number_of_sentences = data.shape[0]
    for i in range(0, number_of_sentences):
        # print(list(cont.expand_texts([data['Tweet_text'][i]]))[-1])
        data['Tweet_text'][i] = list(cont.expand_texts([data['Tweet_text'][i]]))[-1]
        # data['Tweet_text'] = data['Tweet_text'].str.lower()
    print('Contractions finished')


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

def show_missing_words(list_of_not_found_words: List[str]):
    qqq = list_of_not_found_words
    qqq = list(filter(lambda x: x.find('#') == -1 & x.find('...') == -1 & x.find('..') == -1, list_of_not_found_words))

    print(qqq)
    return qqq

def translate_sentence_to_vectors_without_save(data: DataFrame, model: Word2VecKeyedVectors,
                                  output_filename: str, label_encoder: LabelEncoder,
                                  onehot_encoder: OneHotEncoder) -> DataFrame:
    # silence warnings
    pd.set_option('mode.chained_assignment', None)

    series_of_values_asc = data['Tweet_text'].str.len().sort_values(ascending=False)
    list_of_not_found_words: List[str] = []
    for i in range(0, data.shape[0]):
        list_of_vectors = []

        # add postags
        # print(i)
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

    show_missing_words(list_of_not_found_words)
    return data
    # data.to_json(output_filename)
    # return list_of_not_found_words

def translate_sentence_to_vectors_without_save_with_elmo(data: DataFrame, elmo,
                                                         output_filename: str, label_encoder: LabelEncoder,
                                                         onehot_encoder: OneHotEncoder) -> DataFrame:
    # silence warnings
    pd.set_option('mode.chained_assignment', None)
    list_of_not_found_words: List[str] = []
    #data.to_numpy()[:,1]
    list_for_elmo = []
    for tokenized_sentence in data.to_numpy()[:, 1]:
        merged_sentence = ' '.join(tokenized_sentence)
        list_for_elmo.append(merged_sentence)

    print('get elmos embedings')

    list_splited_into_batches = list(divide_chunks(list_for_elmo, 50))
    number_of_sentances = len(list_for_elmo)
    longest_sentance = len(max(list_for_elmo, key=len))
    vector_of_words_for_whole_dataset = np.zeros(shape=(number_of_sentances, longest_sentance, 1024))

    list_count = 0
    tmp = Counter([len(eee) for eee in list_for_elmo])
    for small_list in list_splited_into_batches:
        vector_of_words_for_batch = elmo_vectors(elmo, small_list)
        for single_sentence in vector_of_words_for_batch:
            result = np.zeros(shape=(longest_sentance, 1024))
            result[:single_sentence.shape[0], :single_sentence.shape[1]] = single_sentence
            # Counter(count_of_list_of_sentences)
            vector_of_words_for_whole_dataset[list_count] = result
            list_count = list_count + 1
    print('elmos embedings recieved')

    for i in range(0, data.shape[0]):
        list_of_vectors = []

        # add postags
        # print(i)
        list_of_tokens_in_sentance = data['Tweet_text'][i]
        list_of_tuples_of_tokens_and_tags = nltk.pos_tag(list_of_tokens_in_sentance)
        list_of_tags_in_order_in_sentance = [i[1] for i in list_of_tuples_of_tokens_and_tags]

        list_of_tags_in_order_in_sentance_in_numeric_labels = label_encoder.transform(list_of_tags_in_order_in_sentance)
        list_of_tags_in_order_in_sentance_in_numeric_labels = list_of_tags_in_order_in_sentance_in_numeric_labels.reshape(len(list_of_tags_in_order_in_sentance_in_numeric_labels), 1)
        list_of_tags_in_order_in_sentance_in_onehot_encoded = onehot_encoder.transform(list_of_tags_in_order_in_sentance_in_numeric_labels)

        count = 0
        # print(data['Tweet_text'][i])
        array = data['Tweet_text'][i]
        # vector_of_words = elmo_vectors(elmo, array)
        vector_of_words = vector_of_words_for_whole_dataset[i]
        for word_token in array:
            try:
                # vector_from_word = elmo.get_vector(word_token)
                # vector_from_word = elmo_vectors(elmo, [word_token])
                vector_from_word = vector_of_words[count]
                # add postags
                vector_onehot_encoded = list_of_tags_in_order_in_sentance_in_onehot_encoded[count]
                list_of_vectors.append(np.append(vector_from_word, vector_onehot_encoded))
            except KeyError:
                list_of_not_found_words.append(word_token)

            count = count + 1

        data['Tweet_text'][i] = list_of_vectors

    print('################')
    print('creating vectors finished')
    return data


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

