from detection.create_models import get_all_models_gpu, get_all_models_cpu
from detection.data_inputs import split_data_sets, get_data_for_network

from detection.main_functions import trail_all
from typing import List
from keras import Sequential

# constance variables
max_sentence_length = 10
# len_of_vector_embeddings = 50
# len_of_vector_embeddings = 200
len_of_vector_embeddings = 300
postags_length = 46
total_length = len_of_vector_embeddings + postags_length
global_path_to_results = "results2_ft_merged/"

XXXX, YYYY = get_data_for_network(total_length, max_sentence_length, 'both')
# XXXX,YYYY = get_data_for_network(total_length, max_sentence_length, 'red')
# XXXX,YYYY = get_data_for_network(total_length, max_sentence_length, 'one')

X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data_sets(XXXX, YYYY)

models: List[Sequential] = get_all_models_cpu(total_length, max_sentence_length)

trail_all(models, global_path_to_results, X_train, X_val, X_test, Y_train, Y_val,
          Y_test)
