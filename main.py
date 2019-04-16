############
############
############
############
############

import numpy as np
import pandas as pd
from gensim.models import word2vec
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# load the Stanford GloVe model
filename = 'word2vec.txt'
# model = KeyedVectors.load_word2vec_format(filename, binary=False)

# calculate: (king - man) + woman = ?
# result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# print(result)

testData = "Sweet United Nations video. Just in time for Christmas. #imagine #NoReligion  http://t.co/fej2v3OUBR"
print(word_tokenize(testData))



# data = pd.read_csv('SemEval2018-T3-train-taskA.txt', sep="\t")
data = pd.read_csv('test_dane.txt', sep="\t")


print(list(data.columns.values))
print(data['Tweet_index'].values)
print(data['Label'].values)
print(data['Tweet_text'].values)
print("Aaaaaaa")




