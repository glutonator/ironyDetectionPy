from detection.data_inputs import give_data, give_data_reddit
from detection.main_functions import trail_all

max_sentence_length = 40
# len_of_vector_embeddings = 50
# len_of_vector_embeddings = 200
len_of_vector_embeddings = 300
postags_length = 46
total_length = len_of_vector_embeddings + postags_length

X_train, X_val, X_test, Y_train, Y_val, Y_test = give_data_reddit(total_length, max_sentence_length)
global_path_to_results = "results2_ft_reddit/"

trail_all(global_path_to_results, total_length, max_sentence_length, X_train, X_val, X_test, Y_train, Y_val, Y_test)
