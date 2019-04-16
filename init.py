

########################################################
# konwertowanie glove do formatu rozpoznawanego przez gensim
########################################################
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.twitter.27B.25d.txt'
word2vec_output_file = 'word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)
#######################################################
