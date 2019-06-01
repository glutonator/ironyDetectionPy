from typing import List, Any, Union

import nltk
import pandas as pd
from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk import TweetTokenizer
from pandas import DataFrame
from pandas.core.arrays import ExtensionArray

from pycontractions import Contractions


def func(model):
    # calculate: (king - man) + woman = ?
    result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    print(result)


def clean_messages(data: DataFrame, model: Word2VecKeyedVectors):
    # remove urls
    data['Tweet_text'] = data['Tweet_text'].str \
        .replace('(http|ftp|https)://([\\w_-]+(?:(?:\\.[\\w_-]+)+))([\\w.,@?^=%&:/~+#-]*[\\w@?^=%&/~+#-])?', '', regex=True)

    # remove nicks
    data['Tweet_text'] = data['Tweet_text'].str.replace('@[A-Za-z0-9_]+', '', regex=True)

    # remove hashtags
    # data['Tweet_text'] = data['Tweet_text'].str.replace('\s([#][\w_-]+)', '', regex=True)
    # remove hastags also from the begging of sentence
    data['Tweet_text'] = data['Tweet_text'].str.replace('([#][\\w_-]+)', '', regex=True)

    #remove "|"
    data['Tweet_text'] = data['Tweet_text'].str.replace('|', ' ', regex=False)


    # TODO: dodać tutaj to rozpoznawanie modelu z contractions i konwersje do długich form
    cont = Contractions(kv_model=model)

    number_of_sentences = data.shape[0]
    for i in range(0, number_of_sentences):
        # print(list(cont.expand_texts([data['Tweet_text'][i]]))[-1])
        data['Tweet_text'][i] = list(cont.expand_texts([data['Tweet_text'][i]]))[-1]
        # data['Tweet_text'] = data['Tweet_text'].str.lower()

    # convert to lowercase
    data['Tweet_text'] = data['Tweet_text'].str.lower()


    # TODO: dodać lemingi/streming by uciąć 's jeśli bedize z tym problem

    # remove punctuation
    # data['Tweet_text'] = data['Tweet_text'].str.translate(str.maketrans("", ""), str.punctuation)
    # data['Tweet_text'] = data['Tweet_text'].str.translate(string.punctuation)

    # TODO:  KeyError: "word 'erv's' not in vocabulary"
    # data['Tweet_text'] = data['Tweet_text'].str.replace(rf'[{string.punctuation}]', '')


def tokenize_data_test(model):
    cont = Contractions(kv_model=model)
    # cont = Contractions(api_key="glove-twitter-25")
    # tmp = list(cont.expand_texts(["I would like to know how I had done that!",
    # tmp = next(cont.expand_texts(["I would like to know how I had done that!",
    # tmp = next(cont.expand_texts("I would like to know how I had done that!"))
    # tmp = list(cont.expand_texts(["I'd like to know how I'd done that!"]))
    # tmp = list(cont.expand_texts(["He's rumored to have talked to Erv's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)"]))
    # tmp = list(cont.expand_texts(["Gregor's rumored to have talked to Erv's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)"]))
    # tmp = list(cont.expand_texts(["Gregor's bored."]))
    # print(tmp)

    # tknzr = TweetTokenizer()
    tknzr222 = TweetTokenizer(strip_handles=True)
    # tknzr333 = TweetTokenizer(strip_handles=True, reduce_len=True)
    # testData = "Sweet United Nations video. Just in time for Christmas. #imagine #NoReligion  http://t.co/fej2v3OUBR"
    # testData = "@mrdahl87 We are rumored to have talked to car's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)"
    # testData = "@mrdahl87 We are rumored to have talked to his agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)"
    # testData = "@mrdahl87 We are rumored to have talked to Erv's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)"
    # testData = "@mrdahl87 He's rumored to have talked to Erv's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)"
    # TODO jak sobie poradzić ze słowami nazwa_wlasna + "'s" ??? Odp. Ucinianie "'s"? Bo inaczej to ciezko:D

    testData = "He's bored."

    # print(tknzr.tokenize(testData))
    print(tknzr222.tokenize(testData))
    # print(tknzr333.tokenize(testData))
    temp = tknzr222.tokenize(testData)

    ps = nltk.stem.PorterStemmer()
    for word in temp:
        print(ps.stem(word))

    lemma = nltk.wordnet.WordNetLemmatizer()
    for word in temp:
        print(lemma.lemmatize(word))


def tokenize_data(data):
    tknzr = TweetTokenizer()
    data['Tweet_text'] = data['Tweet_text'].apply(tknzr.tokenize)
    # for i in range(0, data.shape[0]):
    #     print(data['Tweet_text'][i])
    #     data['Tweet_text'][i] = tknzr.tokenize(data['Tweet_text'][i])


def translate_sentence_to_vectors(data: DataFrame, model: Word2VecKeyedVectors, output_filename: str):
    list_of_not_found_words: List[str] = []
    for i in range(0, data.shape[0]):
        # print(data['Tweet_text'][i])
        list_of_vectors = []
        for j in data['Tweet_text'][i]:
            # print(j)
            try:
                list_of_vectors.append(model.get_vector(j))
                # print(model.get_vector(j))
            except KeyError:
                # print(j)
                list_of_not_found_words.append(j)

            data['Tweet_text'][i] = list_of_vectors

    data.to_json(output_filename)
    # data.to_json('vector_test.txt')
    # data.to_csv('vector_test.txt', encoding='utf-8', index=False)
    return list_of_not_found_words


def print_all(data):
    print(list(data.columns.values))
    print(data['Tweet_index'].values)
    print(data['Label'].values)
    print(data['Tweet_text'].values)
    print(data.dtypes)
