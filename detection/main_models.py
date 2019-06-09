import time
from typing import List

from keras import Sequential
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import detection.irony_models as di
from detection.data_inputs import give_data

# init
max_sentence_length = 20
len_of_vector_embeddings = 50
X_train, X_val, X_test, Y_train, Y_val, Y_test = give_data(len_of_vector_embeddings, max_sentence_length)
global_path_to_results = "results/"
ts = str(round(time.time()))

models: List[Sequential] = []
di.baseline_00(X_train, X_val, X_test, Y_train, Y_val, Y_test)

# models.append(di.give_model_10(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_20(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_30(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_40(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_50(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_60(len_of_vector_embeddings, max_sentence_length))
#
# models.append(di.give_model_41(len_of_vector_embeddings, max_sentence_length))
# models.append(di.give_model_61(len_of_vector_embeddings, max_sentence_length))


for model, function_name in models:
    path = global_path_to_results + ts + "_" + function_name + "/"
    di.create_dir(path)
    # di.train_model(model, X_train, X_val, X_test, Y_train, Y_val, Y_test, path)
    di.train_model_learing_rate(model, X_train, X_val, X_test, Y_train, Y_val, Y_test, path, 0.0001)
    di.eval_model(model, X_test, Y_test, path)
    print(function_name)
