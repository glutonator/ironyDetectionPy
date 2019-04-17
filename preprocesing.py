import nltk
from nltk import TweetTokenizer


def func(model):
    # calculate: (king - man) + woman = ?
    result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    print(result)


def clean_messages(data):
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


def tokenize_data_test():
    # tknzr = TweetTokenizer()
    tknzr222 = TweetTokenizer(strip_handles=True)
    # tknzr333 = TweetTokenizer(strip_handles=True, reduce_len=True)
    # testData = "Sweet United Nations video. Just in time for Christmas. #imagine #NoReligion  http://t.co/fej2v3OUBR"
    # testData = "@mrdahl87 We are rumored to have talked to car's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)"
    # testData = "@mrdahl87 We are rumored to have talked to his agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)"
    testData = "@mrdahl87 We are rumored to have talked to Erv's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)"
    # testData = "@mrdahl87 He's rumored to have talked to Erv's agent... and the Angels asked about Ed Escobar... that's hardly nothing    ;)"
    #TODO jak sobie poradzić ze słowami nazwa_wlasna + "'s"

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


def translate_sentence_to_vectors(data, model):
    for i in range(0, data.shape[0]):
        print(data['Tweet_text'][i])
        for j in data['Tweet_text'][i]:
            print(j)
            print(model.get_vector(j))


def print_all(data):
    print(list(data.columns.values))
    print(data['Tweet_index'].values)
    print(data['Label'].values)
    print(data['Tweet_text'].values)
    print(data.dtypes)
