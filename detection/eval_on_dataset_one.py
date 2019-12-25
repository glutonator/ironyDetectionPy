from __future__ import print_function

import sys

import datetime
import os
import time
import numpy as np

from typing import List

from tensorflow_core.python.keras import Sequential

from detection.create_models import get_all_models_gpu
from detection.data_inputs import get_data_from_dataset_one
from detection.main_functions import eval_model_on_dataset_one, eval_f1_on_dataset_one



def eval_on_dataset_one(models: List[Sequential], global_path_to_results,
              X_test, Y_test):
    # ts = str(round(time.time()))
    ts = "1573398849"
    # di_gpu.baseline_00(X_train, X_val, X_test, Y_train, Y_val, Y_test)

    # models: List[Sequential] = get_all_models(total_length, max_sentence_length)

    start = datetime.datetime.now()

    for model, function_name in models:
        path = global_path_to_results + ts + "_" + function_name + "/"
        # create_dir(path)
        # di.train_model(model, X_train, X_val, X_test, Y_train, Y_val, Y_test, path)
        # train_model_learing_rate(model, X_train, X_val, X_test, Y_train, Y_val, Y_test, path, 0.0001)
        # train_model_learing_rate(model, X_train, X_val, X_test, Y_train, Y_val, Y_test, path, 0.001)
        eval_model_on_dataset_one(model, X_test, Y_test, path)
        eval_f1_on_dataset_one(model, X_test, Y_test, path)
        # eval_model_validation(model, X_val, Y_val, path)
        # eval_f1(model, X_test, Y_test, path)
        print(function_name)

    stop = datetime.datetime.now()
    delta = stop - start
    print(delta)




# constance variables
max_sentence_length = 25
# len_of_vector_embeddings = 50
# len_of_vector_embeddings = 200
len_of_vector_embeddings = 300
postags_length = 46
total_length = len_of_vector_embeddings + postags_length
global_path_to_results = "results2_ft_merged/"


XXXX_one, YYYY_one = get_data_from_dataset_one(total_length, max_sentence_length)

models: List[Sequential] = get_all_models_gpu(total_length, max_sentence_length)

eval_on_dataset_one(models, global_path_to_results, XXXX_one, YYYY_one)
