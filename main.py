import numpy as np
import pandas as pd
from gensim.models import word2vec
from gensim.models import KeyedVectors
import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
import string

from pandas import DataFrame

from load_files import load_glove_model, load_input_data
from preprocesing import clean_messages, print_all, tokenize_data, tokenize_data_test, translate_sentence_to_vectors

model = load_glove_model()

# tokenize_data_test(model)
#
data: DataFrame = load_input_data()
#
clean_messages(data, model)
#
tokenize_data(data)

# print(data['Tweet_text'][0])
# print(data.size)
# print(data.shape[0])
# print()
# print(model.get_vector('sweet'))
#
# TODO: ['we', 'are', 'rumored', 'to', 'have', 'talked', 'to', 'erv', 'agent', '...', 'and', 'the', 'angels', 'asked', 'about', 'ed', 'escobar', '...', "that's", 'hardly', 'nothing', ';)']
# TODO: usunąć wielokropek

list_of_not_found_words = translate_sentence_to_vectors(data, model)
#

print("_________________________________")
print(list_of_not_found_words)

# print_all(data)
#
# print("Aaaaaaa")
