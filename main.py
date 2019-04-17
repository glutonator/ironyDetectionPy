import numpy as np
import pandas as pd
from gensim.models import word2vec
from gensim.models import KeyedVectors
import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
import string

from load_files import load_glove_model, load_input_data
from preprocesing import clean_messages, print_all, tokenize_data, tokenize_data_test, translate_sentence_to_vectors

# model = load_glove_model()

tokenize_data_test()

# data = load_input_data()
#
# clean_messages(data)
#
# tokenize_data(data)
#
# print(data['Tweet_text'][0])
# print(data.size)
# print(data.shape[0])
# print()
# print(model.get_vector('sweet'))
#
# translate_sentence_to_vectors(data, model)
#
# print_all(data)

print("Aaaaaaa")
