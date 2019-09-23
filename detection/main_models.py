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
# len_of_vector_embeddings = 50
len_of_vector_embeddings = 200
postags_length = 46
total_length = len_of_vector_embeddings + postags_length

X_train, X_val, X_test, Y_train, Y_val, Y_test = give_data(total_length, max_sentence_length)
global_path_to_results = "results2/"
ts = str(round(time.time()))

models: List[Sequential] = []
di.baseline_00(X_train, X_val, X_test, Y_train, Y_val, Y_test)

models.append(di.give_model_10(total_length, max_sentence_length))
models.append(di.give_model_20(total_length, max_sentence_length))
models.append(di.give_model_30(total_length, max_sentence_length))
models.append(di.give_model_40(total_length, max_sentence_length))
models.append(di.give_model_50(total_length, max_sentence_length))
models.append(di.give_model_60(total_length, max_sentence_length))

models.append(di.give_model_41(total_length, max_sentence_length))
models.append(di.give_model_61(total_length, max_sentence_length))


for model, function_name in models:
    path = global_path_to_results + ts + "_" + function_name + "/"
    di.create_dir(path)
    # di.train_model(model, X_train, X_val, X_test, Y_train, Y_val, Y_test, path)
    di.train_model_learing_rate(model, X_train, X_val, X_test, Y_train, Y_val, Y_test, path, 0.0001)
    di.eval_model(model, X_test, Y_test, path)
    print(function_name)
