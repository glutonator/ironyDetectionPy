from pandas import DataFrame
import tensorflow.compat.v1 as tf


from tensorflow_core.python.keras import Sequential

from detection.create_models import get_all_models_gpu
from detection.data_inputs import split_data_sets, get_data_for_network, get_data_from_dataset_three
from detection.main_detection import prepare_data_for_network

from detection.main_functions import trail_all
from typing import List

# constance variables
max_sentence_length = 25
# len_of_vector_embeddings = 50
# len_of_vector_embeddings = 200
# todo: change back
# len_of_vector_embeddings = 25
len_of_vector_embeddings = 1024
# len_of_vector_embeddings = 300
postags_length = 46
total_length = len_of_vector_embeddings + postags_length
global_path_to_results = "results3_ft_merged/"

# XXXX, YYYY = get_data_for_network(total_length, max_sentence_length, 'both')
# XXXX, YYYY = get_data_for_network(total_length, max_sentence_length, 'red')
# XXXX, YYYY = get_data_for_network(total_length, max_sentence_length, 'one')


# to jest ważne, bez tego nie działa z tf v2.0
tf.disable_eager_execution()

#todo: check if correct
#create with noraml model
# data: DataFrame = prepare_data_for_network('model')
#create with elmo
data: DataFrame = prepare_data_for_network('elmo')


XXXX, YYYY = get_data_from_dataset_three(data, total_length, max_sentence_length)

# XXXX, YYYY = get_data_for_network(total_length, max_sentence_length, 'three')
#
#
X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data_sets(XXXX, YYYY)

models: List[Sequential] = get_all_models_gpu(total_length, max_sentence_length)

trail_all(models, global_path_to_results, X_train, X_val, X_test, Y_train, Y_val,
          Y_test)
