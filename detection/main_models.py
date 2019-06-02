import time
from typing import List

from keras import Sequential

import detection.irony_models as di
from detection.data_inputs import give_data

# init
max_sentence_length = 15
X_train, X_test, Y_train, Y_test = give_data(max_sentence_length)
len_of_vector_embeddings = 25
global_path_to_results = "results/"
ts = str(round(time.time()))

models: List[Sequential] = []

# exp 1
# models.append(di.give_model_10(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_20(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_30(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_40(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_50(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_60(len_of_vector_embeddings, max_sentence_length))

# exp2
# models.append(di.give_model_40(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_41(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_60(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_61(len_of_vector_embeddings, max_sentence_length))

# exp3

# models: List[Sequential] = []
# models.append(di.give_model_61(len_of_vector_embeddings, max_sentence_length))
#
# for model, function_name in models:
#     ts = str(round(time.time()))
#     path = global_path_to_results + ts + "_" + function_name + "/"
#     di.create_dir(path)
#     di.train_model_learing_rate(model, X_train, X_test, Y_train, Y_test, path, 0.01)
#     di.eval_model(model, X_test, Y_test, path)
#     print(function_name)
#
# models: List[Sequential] = []
# models.append(di.give_model_61(len_of_vector_embeddings, max_sentence_length))
#
# for model, function_name in models:
#     ts = str(round(time.time()))
#     path = global_path_to_results + ts + "_" + function_name + "/"
#     di.create_dir(path)
#     di.train_model(model, X_train, X_test, Y_train, Y_test, path)
#     di.eval_model(model, X_test, Y_test, path)
#     print(function_name)
#
# models: List[Sequential] = []
# models.append(di.give_model_61(len_of_vector_embeddings, max_sentence_length))
#
# for model, function_name in models:
#     ts = str(round(time.time()))
#     path = global_path_to_results + ts + "_" + function_name + "/"
#     di.create_dir(path)
#     di.train_model_learing_rate(model, X_train, X_test, Y_train, Y_test, path, 0.0001)
#     di.eval_model(model, X_test, Y_test, path)
#     print(function_name)


# models.append(di.give_model_61(len_of_vector_embeddings, max_sentence_length))


for model, function_name in models:
    ts = str(round(time.time()))
    path = global_path_to_results + ts + "_" + function_name + "/"
    di.create_dir(path)
    di.train_model(model, X_train, X_test, Y_train, Y_test, path)
    di.eval_model(model, X_test, Y_test, path)
    print(function_name)
