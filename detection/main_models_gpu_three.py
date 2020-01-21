from gensim.models.keyedvectors import Word2VecKeyedVectors
from pandas import DataFrame
import tensorflow.compat.v1 as tf


from tensorflow_core.python.keras import Sequential

from detection.create_models import get_all_models_gpu
from detection.data_inputs import split_data_sets, get_data_from_dataset_three
from detection.main_detection import prepare_data_for_network

from detection.main_functions import trail_all
from typing import List, Tuple

# constance variables
# todo: change back
# max_sentence_length = 5
max_sentence_length = 50
# max_sentence_length = 25
# len_of_vector_embeddings = 50
# len_of_vector_embeddings = 200
# todo: change back
# len_of_vector_embeddings = 25
# len_of_vector_embeddings = 1024
len_of_vector_embeddings = 300
postags_length = 46
# with_postags = False
with_postags = True
#todo: teraz trzeba elmo ogarnać

if with_postags == True:
    total_length = len_of_vector_embeddings + postags_length
else:
    total_length = len_of_vector_embeddings

global_path_to_results = "results7_ft_merged/"

# XXXX, YYYY = get_data_for_network(total_length, max_sentence_length, 'both')
# XXXX, YYYY = get_data_for_network(total_length, max_sentence_length, 'red')
# XXXX, YYYY = get_data_for_network(total_length, max_sentence_length, 'one')


# to jest ważne, bez tego nie działa z tf v2.0
tf.disable_eager_execution()

#todo: chnge hastags to sth like -> model.similar_by_word('<hashtag>')


#todo: check if correct
#create with noraml model
data, model = prepare_data_for_network(max_sentence_length, with_postags, 'model')

# data_test, modelXXX = prepare_data_for_network(max_sentence_length, with_postags, 'model', model,
#                                                preprocessed_file_to_test='preprocessed_data_fastText_dataset_one.txt')

#create with elmo
# data, model = prepare_data_for_network(max_sentence_length, with_postags, 'elmo')
# model = None
# data_test, modelXXX = prepare_data_for_network(max_sentence_length, with_postags, 'elmo', model,
#                                                preprocessed_file_to_test='preprocessed_data_fastText_dataset_one.txt')


XXXX, YYYY = get_data_from_dataset_three(data, total_length, max_sentence_length)

# XXXX_test, YYYY_test = get_data_from_dataset_three(data_test, total_length, max_sentence_length)

# XXXX, YYYY = get_data_for_network(total_length, max_sentence_length, 'three')
#
#
X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data_sets(XXXX, YYYY)
# X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data_sets(XXXX, YYYY, XXXX_test, YYYY_test)

models: List[Sequential] = get_all_models_gpu(total_length, max_sentence_length)

trail_all(models, global_path_to_results, X_train, X_val, X_test, Y_train, Y_val,
          Y_test)
