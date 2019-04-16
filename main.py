import numpy as np
import pandas as pd
from gensim.models import word2vec
from gensim.models import KeyedVectors
import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
import string

# load the Stanford GloVe model
filename = 'word2vec.txt'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

# calculate: (king - man) + woman = ?
# result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# print(result)

tknzr = TweetTokenizer()
# testData = "Sweet United Nations video. Just in time for Christmas. #imagine #NoReligion  http://t.co/fej2v3OUBR"
testData = "@mrdahl87 We are rumored to have talked to Erv's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)"
testData = "@mrdahl87 He's rumored to have talked to Erv's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)"

print(tknzr.tokenize(testData))

# data = pd.read_csv('SemEval2018-T3-train-taskA.txt', sep="\t")
data = pd.read_csv('test_dane.txt', sep="\t")

# remove urls
data['Tweet_text'] = data['Tweet_text'].str \
    .replace('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', regex=True)

# remove nicks
data['Tweet_text'] = data['Tweet_text'].str.replace('@[A-Za-z0-9]+', '', regex=True)

# remove hashtags
data['Tweet_text'] = data['Tweet_text'].str.replace('\s([#][\w_-]+)', '', regex=True)


# convert to lowercase
data['Tweet_text'] = data['Tweet_text'].str.lower()

# remove punctuation
# data['Tweet_text'] = data['Tweet_text'].str.translate(str.maketrans("", ""), str.punctuation)
# data['Tweet_text'] = data['Tweet_text'].str.translate(string.punctuation)

# TODO:  KeyError: "word 'erv's' not in vocabulary"
# data['Tweet_text'] = data['Tweet_text'].str.replace(rf'[{string.punctuation}]', '')

data['Tweet_text'] = data['Tweet_text'].apply(tknzr.tokenize)
# data['Tweet_text'] = data['Tweet_text'].

print(data['Tweet_text'][0])
print(data.size)
print(data.shape[0])
print()
print(model.get_vector('sweet'))
for i in range(0, data.shape[0]):
    print(data['Tweet_text'][i])
    for j in data['Tweet_text'][i]:
        print(j)
        print(model.get_vector(j))


# for label, content in data.iteritems():
#     print('label:', label)
#     print('content:', content, sep='\n')

# print(list(data.columns.values))
# print(data['Tweet_index'].values)
# print(data['Label'].values)
# print(data['Tweet_text'].values)
# print(data.dtypes)

print("Aaaaaaa")
