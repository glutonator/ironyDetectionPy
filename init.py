
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

########################################################
# konwertowanie glove do formatu rozpoznawanego przez gensim
########################################################
from gensim.scripts.glove2word2vec import glove2word2vec
# glove_input_file = 'glove.twitter.27B.50d.txt'
glove_input_file = 'glove.twitter.27B.200d.txt'
# word2vec_output_file = 'word2vec_50.txt'
word2vec_output_file = 'word2vec_200.txt'
glove2word2vec(glove_input_file, word2vec_output_file)
#######################################################
